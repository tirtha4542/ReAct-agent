import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client
from langchain.tools import tool
import requests

# Import from the classic package
from langchain_classic.agents import AgentExecutor, create_react_agent

load_dotenv()

# 1. Initialize Model
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

# 2. Initialize Search Tool
search_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3,
    return_source_documents=True
)
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a given city using the Weatherstack API."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"
    response = requests.get(url)    
    DATA = response.json()
    if "current" in DATA:
        temperature = DATA["current"]["temperature"]
        weather_description = DATA["current"]["weather_descriptions"][0]
        return f"The current temperature in {city} is {temperature}°C with {weather_description}."

tools = [search_tool, get_weather]

# 3. Pull the ReAct prompt from LangSmith hub
client = Client()
prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

# 4. Create the classic ReAct agent
agent = create_react_agent(llm, tools, prompt)

# 5. Wrap the agent inside AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# Example invocation:
response = agent_executor.invoke({"input": "what is the capital of bangladesh and what is the current weather in this city?"})
print(response["output"])