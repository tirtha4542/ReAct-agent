import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from langsmith import Client
from langchain.tools import tool
import requests

from langchain_classic.agents import AgentExecutor, create_react_agent

load_dotenv()

st.set_page_config(page_title="ReAct Agent Chat", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# API keys — loaded silently from .env, never shown in the UI
# ---------------------------------------------------------------------------
mistral_key = os.getenv("MISTRAL_API_KEY", "")
tavily_key = os.getenv("TAVILY_API_KEY", "")
weatherstack_key = os.getenv("WEATHERSTACK_API_KEY", "")
langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")

# ---------------------------------------------------------------------------
# Sidebar — just app settings, no keys
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    show_thoughts = st.toggle("Show agent's reasoning (thoughts/actions)", value=True)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city using the Weatherstack API."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    response = requests.get(url)
    data = response.json()
    if "current" in data:
        temperature = data["current"]["temperature"]
        weather_description = data["current"]["weather_descriptions"][0]
        return f"The current temperature in {city} is {temperature}°C with {weather_description}."
    return f"Could not fetch weather for {city}."


# ---------------------------------------------------------------------------
# Agent construction (cached so it's built only once per session/key-set)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the agent...")
def build_agent_executor(_mistral_key, _tavily_key):
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, max_retries=2)

    search_tool = TavilySearchResults(
        api_key=_tavily_key, max_results=3, return_source_documents=True
    )

    tools = [search_tool, get_weather]

    client = Client()
    prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------
st.title("🤖 ReAct Agent — Search + Weather")
st.caption("Powered by Mistral, Tavily search, and Weatherstack, via a LangChain ReAct agent.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything, e.g. 'What is the capital of Bangladesh and its weather?'")

if user_input:
    if not mistral_key or not tavily_key:
        st.error("Missing MISTRAL_API_KEY or TAVILY_API_KEY. Please set them in your .env file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            agent_executor = build_agent_executor(mistral_key, tavily_key)

            if show_thoughts:
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.invoke(
                    {"input": user_input}, {"callbacks": [st_callback]}
                )
            else:
                with st.spinner("Thinking..."):
                    response = agent_executor.invoke({"input": user_input})

            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = f"⚠️ Something went wrong: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})