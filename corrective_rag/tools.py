from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
import logging
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

class Tools():
    """
    A utility class to create and manage various search tools for querying information
    from DuckDuckGo, Tavily, Wikipedia, and Arxiv.

    Attributes:
        logger (Logger): Logger instance for debugging and informational logs.
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def duck_duck_search_tool(self):
        """
        Creates a DuckDuckGo search tool using the API wrapper.
        The tool is configured to fetch a maximum of 3 results, focusing on news sources.

        Returns:
            DuckDuckGoSearchResults: A tool for executing DuckDuckGo searches.
        """
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3,source='news')
        search = DuckDuckGoSearchResults(api_wrapper=wrapper)
        self.logger.info("Created a duck duck go web search tool successfully.")
        return search
    
    def tavily_search_tool(self):
        """
        Creates a Tavily search tool for advanced web search queries.
        Configured to return a maximum of 3 results with a token limit of 250 for responses.

        Returns:
            TavilySearchResults: A tool for performing Tavily searches.
        """
        tv_search = TavilySearchResults(
            
            max_results=3,
            search_depth='advanced',
            max_tokens=250
        )
        self.logger.info("Created a tavily web search tool successfully.")
        return tv_search
    
    def wiki_search_tool(self):
        """
        Creates a Wikipedia search tool using the Wikipedia API wrapper.
        Configured to fetch a maximum of 3 results, limiting document content to 300 characters.

        Returns:
            WikipediaQueryRun: A tool for querying information from Wikipedia.
        """
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=300)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        self.logger.info("Created a wikipedia search tool successfully.")
        return wiki_tool
    
    def arxiv_search_tool(self):
        """
        Creates an Arxiv search tool using the Arxiv API wrapper.
        Configured to fetch a maximum of 3 results, limiting document content to 300 characters.

        Returns:
            ArxivQueryRun: A tool for querying research papers from Arxiv.
        """
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=3,doc_content_chars_max=300)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        self.logger.info("Created a arxiv search tool successfully.")
        return arxiv_tool

