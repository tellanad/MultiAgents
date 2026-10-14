from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent
from langgraph.prebuilt import create_react_agent


# Initialize Tavily client backends
tavily_advanced = TavilySearch(max_results=5, search_depth="advanced")
tavily_basic = TavilySearch(max_results=3, search_depth="basic")







