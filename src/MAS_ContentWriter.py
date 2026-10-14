from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from deepagents import create_deep_agent
from langgraph.prebuilt import create_react_agent
import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERP_API_KEY= os.getenv("SERP_API_KEY")

# Initialize Tavily client backends
tavily_advanced = TavilySearch(max_results=5, search_depth="advanced")
tavily_basic = TavilySearch(max_results=3, search_depth="basic")


@tool
def research_tool(topic_or_hypothesis: str) -> str:
    """Run an in-depth research sweep over high-authority sources, academic papers,
    and long-form publications.

    COST NOTICE: This is an expensive and heavy tool call. Only invoke this ONCE
    or TWICE per session when deep context or structured evidence gathering is required.
    Do NOT use for surface lookups or simple facts.
    """
    response = tavily_advanced.invoke({"query": topic_or_hypothesis})
    return str(response)


@tool
def internet_search_tool(query: str) -> str:
    """Run a fast, lightweight search across the public web.

    Use this freely and repeatedly for quick fact checks, SEO trends, keyword
    discovery, company updates, or specific single-fact lookups.
    """
    response = tavily_basic.invoke({"query": query})
    return str(response)





