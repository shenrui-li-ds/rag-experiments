import streamlit as st
import ollama
import httpx
import requests

# @st.cache_data
def get_openai_models(OPENAI_API_KEY) -> list:
    api_provider = st.session_state["api_provider"]
    if api_provider == "OpenAI":
        url = "https://api.openai.com/v1/models"
        headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            return sorted([i['id'] for i in response.json()['data'] if 'gpt' in i['id']])
        else:
            print(f"Failed to fetch OpenAI models, status code: {response.status_code}")
            return []

def get_mistral_models(MISTRAL_API_KEY) -> list:
    api_provider = st.session_state["api_provider"]
    if api_provider == "Mistral":
        url = "https://api.mistral.ai/v1/models"
        headers = {'Authorization': f'Bearer {MISTRAL_API_KEY}'}
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            return sorted([i['id'] for i in response.json()['data']])
        else:
            print(f"Failed to fetch Mistral models, status code: {response.status_code}")
            return []

# @st.cache_data
def get_ollama_models() -> list:
    api_provider = st.session_state["api_provider"]
    if api_provider == "Ollama":
        try:
            # Return the list of available local models if connected successfully
            return sorted([m['name'] for m in ollama.list()['models']])
        except httpx.ConnectError as e:
            print(f"Connection error: {e}")
            st.warning(' You need to run Ollama in the background!', icon='⚠️')
            # Handle the connection error or return a fallback value
            return []
        except Exception as e:
            # Catch other exceptions to avoid failing silently
            print(f"An unexpected error occurred: {e}")
            return []

def check_ollama_connection() -> bool:
    url = 'http://localhost:11434/'
    try:
        response = requests.get(url)
        # If the server is reachable and returns a status code of 200, return True
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        # If a ConnectionError is caught, print a message or handle it as needed
        print("Failed to connect to the Ollama backend.")
    # Return False if the server is not reachable or any other error occurs
    return False

def side_bar():
    with st.sidebar:
        st.title("Navigation")
        # Page options
        pages = ["RAG"]
        # pages = ["Chat", "Prompt", "RAG", "Multi-Agents"]
        # Sidebar navigation
        selected_page = st.sidebar.radio("Select a page", pages)

        # Model selection
        st.subheader('Models and parameters')
        api_provider = st.selectbox("Select API Provider", ("OpenAI", "Anthropic", "Ollama", "Google", "Mistral"), key="main_api_provider")
        your_api_key = st.text_input("Your API Key", value=None, type="password")

        # Stateful "Save" button    
        if 'api_saved' not in st.session_state:
            st.session_state.api_saved = False
        
        def save_api_settings():
            st.session_state.api_saved = True
        
        st.button('Save', on_click=save_api_settings)

        if st.session_state.api_saved:
            if (api_provider == "Ollama" and check_ollama_connection()) or (api_provider != "Ollama" and your_api_key):
                st.session_state.api_saved = True
                st.session_state["api_provider"] = api_provider
                st.session_state.secrets = {
                    "your_api_key": your_api_key, 
                }
                st.success(" Proceed to entering your prompt message!", icon='👉')
            else:
                st.warning(' Please enter your credentials!', icon='⚠️')
        else:
            st.warning(' Please enter your credentials!', icon='⚠️')

        models = {
            "OpenAI": get_openai_models(your_api_key) if your_api_key else [],
            "Anthropic": ["claude-2", "claude-2.1", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
            "Ollama": get_ollama_models(),
            "Google": ["gemini-pro"],
            "Mistral": get_mistral_models(your_api_key) if your_api_key else [],
            # "Mistral": ["open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b", "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        }
        
        if "api_provider" in st.session_state and (st.session_state["api_provider"] is not None) and "your_api_key" in st.session_state.secrets:
            api_provider = st.session_state["api_provider"]
            provided_models = models.get(api_provider, [])
            selected_model = st.selectbox("Select a model", provided_models)
            # Save the selected model in session state
            st.session_state["selected_model"] = selected_model
            # Displaying the current model settings
            current_settings = f"[{api_provider}] - [{selected_model}]"
            st.caption(f"Current Model: {current_settings}")
        else:
            st.caption("No API provider or model selected.")

        temperature = st.sidebar.slider('temperature', min_value=0.0, max_value=2.0, value=0.1, step=0.1)
        top_p = st.sidebar.slider('top_p', min_value=0.0, max_value=1.0, value=0.9, step=0.01)
        max_tokens = st.sidebar.slider('max_tokens', min_value=512, max_value=128_000, value=512, step=256)
        st.session_state['temperature'] = temperature
        st.session_state['top_p'] = top_p
        st.session_state['max_tokens'] = max_tokens

        st.write("\n")

        # # Button to clear chat history
        # if st.button("Clear Chat History"):
        #     st.session_state.messages = []

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
            # st.session_state.messages = []
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    return selected_page
