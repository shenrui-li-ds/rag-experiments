import streamlit as st
import os
import tempfile
import time
import tiktoken
# import logging

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from flashrank import Ranker
from typing import Dict


from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader, 
    PythonLoader,
    TextLoader, 
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    # UnstructuredImageLoader,
    UnstructuredMarkdownLoader, 
    UnstructuredWordDocumentLoader, 
)

from langchain.retrievers import EnsembleRetriever
from pydantic import BaseModel, Field

import warnings
warnings.filterwarnings('ignore')

# Custom loader class to set utf-8 encoding
class CustomTextLoader(TextLoader):
    def __init__(self, file_path, encoding='utf-8'):
        super().__init__(file_path)
        self.encoding = encoding


class CustomPromptTemplate(BaseModel):
    # Now context is defined as a list of strings
    context: list
    input: str

    @classmethod
    def from_template(cls, context: list, input: str):
        return cls(context=context, input=input)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )

def get_loader(file_path):
    # Map of file extensions to their corresponding loader classes
    loaders = {
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".html": UnstructuredHTMLLoader,
        ".json": JSONLoader,
        # ".jpg": UnstructuredImageLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        # ".png": UnstructuredImageLoader,
        ".py": PythonLoader,
        ".txt": CustomTextLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    # Retrieve the loader class based on the file extension
    loader_class = loaders.get(file_extension)

    if not loader_class:
        raise ValueError(f"Unsupported file type: \"{file_extension}\". Please remove this file to proceed.")
    
    return loader_class

def get_semantic_chunker(api_provider, api_key=""):
    # Mapping of providers to their API key parameter names, embedding classes, and model names
    provider_config = {
        "Mistral": {
            "api_key_param": "mistral_api_key",
            "embedding_class": MistralAIEmbeddings,
            "model_name": "mistral-embed",
        },
        "OpenAI": {
            "api_key_param": "openai_api_key",
            "embedding_class": OpenAIEmbeddings,
            "model_name": "text-embedding-3-large",
        },
        "Ollama": {
            "embedding_class": OllamaEmbeddings,
            "model_name": "mxbai-embed-large",
        }
    }

    # Check if the provider is supported
    if api_provider not in provider_config:
        raise ValueError(f"Unsupported API provider: {api_provider}")

    # Get the configuration for the selected provider
    config = provider_config[api_provider]

    # Prepare parameters based on whether an API key is needed
    params = {}
    if "api_key_param" in config:
        params[config["api_key_param"]] = api_key

    # Instantiate the embedding model with the appropriate model name and parameters
    embedding_model = config["embedding_class"](model=config["model_name"], **params)
    
    # Assuming SemanticChunker is relevant for all providers
    return SemanticChunker(embedding_model, buffer_size=2, 
                           breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)

def get_sparse_retriever(knowledge, k=20):
    bm25_retriever = BM25Retriever.from_documents(
        knowledge
    )
    bm25_retriever.k = k
    return bm25_retriever

def get_hybrid_retriever(knowledge, embeddings, k=20):
    bm25_retriever = BM25Retriever.from_documents(
        knowledge
    )
    bm25_retriever.k = k
    faiss_vectordb = FAISS.from_documents(knowledge, embeddings)
    faiss_retriever = faiss_vectordb.as_retriever(search_kwargs={"k": 30})
    ensemble_retriever = get_ensemble_retriever(bm25_retriever, faiss_retriever)

    return ensemble_retriever

def get_compression_retriever(mode, api_key, 
                              chat_model_class, chat_model_name, 
                              embedding_model_class, embedding_model_name, 
                              base_retriever):
    # Define model parameters
    api_provider = st.session_state["api_provider"]
    key_param = {
        "Mistral": "mistral_api_key",
        "OpenAI": "openai_api_key",
    }
    # Ensure the provider is supported, else raise an exception
    if api_provider in key_param:
        # Providers that require an API key
        api_param = {key_param[api_provider]: api_key}
    elif api_provider == "Ollama":
        # "Ollama" does not require an API key
        api_param = {}
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")

    if mode == "reranker":
        compressor = FlashrankRerank(client=Ranker, top_n=15, model="ms-marco-MiniLM-L-12-v2")

    elif mode == "llm_filter":
        temperature = st.session_state["temperature"]
        model_params = {
            'temperature': temperature,
        }
        llm = chat_model_class(
            model=chat_model_name, 
            **api_param,
            **model_params,
        )
        compressor = LLMChainFilter.from_llm(llm)

    elif mode == "embeddings_filter":
        embeddings = embedding_model_class(
            model=embedding_model_name, 
            **api_param,
        )
        compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)

    elif mode == "compressor_pipeline":
        embeddings = embedding_model_class(
            model=embedding_model_name, 
            **api_param,
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
        compressor = DocumentCompressorPipeline(
            transformers=[redundant_filter, relevant_filter]
)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever,
    )
    return compression_retriever

def get_multi_query_retriever(chat_model_class, base_retriever, api_key):
    llm = chat_model_class(temperature=0, api_key=api_key)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, include_original=True,
    )
    return multi_query_retriever

def get_ensemble_retriever(retriever1, retriever2):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
    )
    return ensemble_retriever

@st.cache_data
def file_processor(uploaded_files, api_key):
    knowledge = []
    api_provider = st.session_state["api_provider"]
    # Recursive Character Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Semantic Chunking
    # text_splitter = get_semantic_chunker(api_provider, api_key)
    
    for uploaded_file in uploaded_files:
        try:
            loader_class = get_loader(uploaded_file.name)
            # Create tempfile because loader_class takes PathLike objects
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = loader_class(file_path=temp_file_path)
            docs = loader.load()
            # Split text into chunks
            file_knowledge = text_splitter.split_documents(docs)
            for idx, text in enumerate(file_knowledge):
                text.metadata["id"] = idx
            knowledge.extend(file_knowledge)
            os.unlink(temp_file_path)  # Remove the temporary file
        except ValueError as e:
            st.warning(str(e))
            continue

    return knowledge

def calculate_tokens(text, model="text-embedding-3-large"):
    """
    Use tiktoken to count tokens for a given text and model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def embed_documents_in_batches(knowledge, embeddings, max_tokens_per_minute=1000000, model="text-embedding-3-large"):
    """
    Embed the documents in batches to avoid exceeding the OpenAI token rate limit.
    Splits `knowledge` into smaller batches based on token count.
    """
    batch_embeddings = []
    current_batch = []
    total_tokens = 0
    start_time = time.time()

    for document in knowledge:
        # Use tiktoken to calculate the precise token count
        token_count = calculate_tokens(document.page_content, model=model)
        
        if total_tokens + token_count > max_tokens_per_minute:
            # Process current batch
            current_embeddings = embeddings.embed_documents([doc.page_content for doc in current_batch])
            batch_embeddings.extend(current_embeddings)

            # Check time and possibly wait before starting the next batch
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)  # Wait to respect the rate limit
            total_tokens = 0  # Reset token count
            current_batch = []  # Clear batch
            start_time = time.time()  # Reset timer

        # Add document to current batch
        current_batch.append(document)
        total_tokens += token_count

    # Process the last batch
    if current_batch:
        current_embeddings = embeddings.embed_documents([doc.page_content for doc in current_batch])
        batch_embeddings.extend(current_embeddings)

    return batch_embeddings

def generate_chain_of_thought(query, chat_model):
    # This function generates the chain of thought steps in a structured format
    messages = [
        (
            "system", 
            "You are a helpful assistant that breaks down questions into specific sub-questions for better retrieval."
        ),
        (
            "human", 
            f"The user asked: '{query}'. Break down this question into key components "
            "and respond with a numbered list of sub-questions. Format each sub-question "
            "on a new line as follows:\n"
            "1. Sub-question 1\n"
            "2. Sub-question 2\n"
            "..."
        )
    ]
    
    # Use the invoke method to generate the response
    ai_msg = chat_model.invoke(messages)
    chain_of_thought = ai_msg.content

    return chain_of_thought

def base_rag_chain(knowledge, user_query, api_key, 
                   embedding_model_class, embedding_model_name, 
                   chat_model_class, chat_model_name):
    # Define the embedding model
    api_provider = st.session_state["api_provider"]
    key_param = {
        "Mistral": "mistral_api_key",
        "OpenAI": "openai_api_key",
    }
    # Ensure the provider is supported, else raise an exception
    if api_provider in key_param:
        api_param = {key_param[api_provider]: api_key}
    elif api_provider == "Ollama":
        api_param = {}
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")
    
    embeddings = embedding_model_class(
        model=embedding_model_name, 
        **api_param,
    )

    # Define the chat model
    temperature = st.session_state["temperature"]
    top_p = st.session_state["top_p"] if st.session_state["top_p"] else 0.0
    model_params = {
        'temperature': temperature,
        'top_p':top_p,
    }
    chat_model = chat_model_class(
        model=chat_model_name, 
        **api_param,
        **model_params,
    )

    # Use the chain of thought to guide the retrieval
    chain_of_thought = generate_chain_of_thought(user_query, chat_model)
    
    # Create the vector DB 
    # Embed documents in batches to avoid exceeding token rate limits
    batch_embeddings = embed_documents_in_batches(knowledge, embeddings)
    # Combine texts (from `knowledge`) with corresponding embeddings (from `batch_embeddings`)
    text_embeddings = [(doc.page_content, embedding) for doc, embedding in zip(knowledge, batch_embeddings)]
    # Now pass the (text, embedding) pairs to FAISS
    faiss_vectordb = FAISS.from_embeddings(text_embeddings, embeddings)
    # Create FAISS vector store with the embedded documents
    # faiss_vectordb = FAISS.from_embeddings(batch_embeddings, knowledge)
    # faiss_vectordb = FAISS.from_documents(knowledge, embeddings)
    faiss_retriever = faiss_vectordb.as_retriever(search_kwargs={"k": 15})
    # Aggregate all documents gathered from the chain of thoughts
    sub_queries = [
        line.strip() for line in chain_of_thought.split('\n') if line.strip().startswith(tuple('123456789'))
    ]
    retrieved_docs = []
    for sub_query in sub_queries:
        # Remove the numbering for each sub-query
        clean_query = sub_query.split('.', 1)[-1].strip()
        # Retrieve documents for each sub-query
        docs = faiss_retriever.get_relevant_documents(clean_query)
        retrieved_docs.extend(docs)
        
    unique_docs = list({doc.page_content: doc for doc in retrieved_docs}.values())

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat_model, question_answering_prompt)
    
    # document_chain = create_stuff_documents_chain(chat_model, prompt)
    # retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
    retriever = faiss_retriever
    def parse_retriever_input(params: Dict):
        return params["messages"][-1].content
    # retrieval_chain = RunnablePassthrough.assign(
    #     context=parse_retriever_input | retriever,
    # ).assign(
    #     answer=document_chain,
    # )

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template(
                (
                    "Given the above conversation, generate a search query to look up in order to get "
                    "information relevant to the conversation. Only respond with the query, nothing else."
                )
            ),
        ]
    )
    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            # If only one message, then we just pass that message's content to retriever
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
        query_transform_prompt | chat_model | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    # Handle follow up conversations with related context
    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    # Only for debugging, print retrieval result
    pretty_print_docs(unique_docs)
    
    # Stream response to Streamlit
    def generate_responses():
        # response = retrieval_chain.invoke(
        #     {
        #         "messages": [
        #             HumanMessage(content=user_query)
        #         ],
        #     }
        # )
        stream = conversational_retrieval_chain.stream(
            {
                "messages": [
                    HumanMessage(content=user_query),
                ],
            }
        )
        # retrieval_chain.stream({"input": user_query})
        for chunk in stream :
            if 'answer' in chunk:
                yield chunk['answer']
            elif 'result' in chunk:
                yield chunk['result']
    response = st.write_stream(generate_responses)
    st.session_state.messages.append({"role": "assistant", "content": response})


def mistral_rag(knowledge, user_query, mistral_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, mistral_api_key, 
                   MistralAIEmbeddings, "mistral-embed", 
                   ChatMistralAI, chat_model_name)

def openai_rag(knowledge, user_query, openai_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, openai_api_key, 
                   OpenAIEmbeddings, "text-embedding-3-large", 
                   ChatOpenAI, chat_model_name)
    
def ollama_rag(knowledge, user_query, ollama_api_key):
    chat_model_name = st.session_state["selected_model"]
    print(chat_model_name)
    base_rag_chain(knowledge, user_query, ollama_api_key, 
                   OllamaEmbeddings, "nomic-embed-text", 
                   ChatOllama, chat_model_name)

def rag_page():
    st.title("Retrieval-Augmented Generation (RAG)")
    st.caption("Only support OpenAI, Mistral and Ollama models at this time.")
    if "api_provider" in st.session_state and "selected_model" in st.session_state:
        api_provider = st.session_state["api_provider"]
        selected_model = st.session_state["selected_model"]
        st.caption((
            f"Current model: [{api_provider}] - [{selected_model}]. "
            "LLM can make mistakes. Consider checking important information."
        ))
    else:
        st.caption((
            "No API provider or model selected. "
            "LLM can make mistakes. Consider checking important information."
        ))
    uploaded_files = st.file_uploader("Upload your file(s)", accept_multiple_files=True)
    rag_api_provider = st.session_state["api_provider"]

    if "your_api_key" not in st.session_state.secrets:
        st.warning(' Please enter your credentials in the side bar first.', icon='⚠️')

    if uploaded_files:
        # knowledge = file_processor(uploaded_files)
        if rag_api_provider == "Mistral":
            mistral_api_key = st.session_state.secrets["your_api_key"]
            knowledge = file_processor(uploaded_files, mistral_api_key)
        elif rag_api_provider == "OpenAI":
            openai_api_key = st.session_state.secrets["your_api_key"]
            knowledge = file_processor(uploaded_files, openai_api_key)
        elif rag_api_provider == "Ollama":
            ollama_api_key = ""
            knowledge = file_processor(uploaded_files, ollama_api_key)
    
        if knowledge:
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
                # st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Enter your query here"):
                with st.chat_message("user"):
                    st.markdown(user_query)
                st.session_state.messages.append({"role": "user", "content": user_query})

                with st.chat_message("assistant"):
                    if rag_api_provider == "Mistral":
                        mistral_api_key = st.session_state.secrets["your_api_key"]
                        with st.spinner("Thinking..."):
                            mistral_rag(knowledge, user_query, mistral_api_key)

                    elif rag_api_provider == "OpenAI":
                        openai_api_key = st.session_state.secrets["your_api_key"]
                        with st.spinner("Thinking..."):
                            openai_rag(knowledge, user_query, openai_api_key)

                    elif rag_api_provider == "Ollama":
                        ollama_api_key = ""
                        with st.spinner("Thinking..."):
                            ollama_rag(knowledge, user_query, ollama_api_key)
                    
                    else:
                        st.warning(' Selected provider is not supported at this time.', icon='⚠️')

        else:
            st.warning(' Failed to parse uploaded document(s).', icon='⚠️')

