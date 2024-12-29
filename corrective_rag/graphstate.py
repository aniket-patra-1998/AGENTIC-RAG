from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
import logging
from langgraph.graph import END,StateGraph,START
from langchain_core.pydantic_v1 import BaseModel,Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()



class Graph():

    """
    A class representing a graph-based workflow to manage question answering using various tools 
    and data sources. It processes a user's query through retrieval, grading, rewriting, 
    and generating answers.
    """

    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        Attributes:
            question: question
            generation: LLM response generation
            web_search_needed: flag of whether to add web search - yes or no
            documents: list of context documents
        """
        question: str
        generation: str
        web_search_needed: str
        documents: List[str]

    class RouteQuery(BaseModel):

        """
        Represents a query routing schema for deciding the appropriate data source or tool.

        Attributes:
            datasource (Literal): Specifies the chosen data source ('web_search', 'wiki_search', 'arxiv_search').
        """

        datasource: Literal['web_search',"wiki_search",'arxiv_search'] = Field(
            ...,
            description='Given a user question choose to route it to wikipedia or web search or arxiv search'
        )

    def __init__(self,similarity_threshold_retriever,doc_grader,
                 question_rewriter,web_search_tool,
                 wiki_search_tool,arxiv_search_tool,qa_rag_chain):
        
        """
        Initializes the Graph instance with required components and logging.

        Args:
            similarity_threshold_retriever: A retriever to fetch documents based on query similarity.
            doc_grader: A grader to assess the relevance of documents.
            question_rewriter: A tool to rewrite queries for better search results.
            web_search_tool: A tool to perform web searches.
            wiki_search_tool: A tool to search Wikipedia.
            arxiv_search_tool: A tool to search the arXiv repository.
            qa_rag_chain: A tool to generate answers using retrieved documents.
        """

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        #self.state = state
        self.similarity_threshold_retriever = similarity_threshold_retriever
        self.document_grader = doc_grader
        self.question_rewriter = question_rewriter
        self.web_search_tool = web_search_tool
        self.wiki_search_tool = wiki_search_tool
        self.arxiv_search_tool = arxiv_search_tool
        self.qa_rag_chain = qa_rag_chain
        self.structured_llm_router = None

    def retrieve(self,state):
        """
        Retrieve documents relevant to the input question.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with retrieved documents.
        """
        question = state['question']
        documents = self.similarity_threshold_retriever.invoke(question)
        return {'documents':documents,'question':question}
    
    def grade_documents(self,state):
        """
        Grade the relevance of retrieved documents and filter irrelevant ones.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with filtered documents and a flag for web search necessity.
        """
        question = state['question']
        documents = state['documents']
        filtered_docs = []
        web_search_needed = 'No'

        if documents:
            for d in documents:
                score = self.document_grader.invoke(
                    {'question':question,'document':d.page_content} 
                )
                grade = score.binary_score
                if grade=='yes':
                    self.logger.info("Document is relevant")
                    filtered_docs.append(d)
                else:
                    self.logger.info("Document is irrelevant")
                    web_search_needed = 'Yes'
                    continue
        else:
            self.logger.info("No documents found")
            web_search_needed = 'Yes'
        return {'documents':filtered_docs,'question':question, 'web_search_needed':web_search_needed}
    
    def rewrite_query(self,state):
        """
        Rewrite the query to improve retrieval results.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with a rewritten question.
        """
        question = state['question']
        documents = state['documents']

        self.logger.info("Rephrasing the query")
        better_question = self.question_rewriter.invoke({'question':question})
        self.logger.info(f"Rephrased query: {better_question} ")

        return {'documents':documents,'question':better_question}
    
    def web_search(self,state):
        """
        Perform a web search using the rewritten query.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with web search results added to documents.
        """
        question = state['question']
        documents = state['documents']

        self.logger.info("Performing web search")

        docs = self.web_search_tool.invoke(question)
        web_results = "\n\n".join([d['content'] for d in docs])
        web_results = Document(page_content=web_results)
        
        documents.append(web_results)

        return {'documents':documents, 'question':question}
    
    def wiki_search(self,state):
        """
        Perform a Wikipedia search using the rewritten query.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with Wikipedia search results added to documents.
        """

        question = state['question']
        documents = state['documents']

        docs = self.wiki_search_tool.invoke({'query':question})
        wiki_results = docs
        wiki_results = Document(page_content=wiki_results)

        documents.append(wiki_results)


        return {"documents": documents,'question':question}
    
    def arxiv_search(self,state):
        """
        Perform an arXiv search using the rewritten query.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with arXiv search results added to documents.
        """

        question = state['question']
        documents = state['documents']

        docs = self.arxiv_search_tool.invoke({'query':question})
        arxiv_results = docs
        arxiv_results = Document(page_content=arxiv_results)

        documents.append(arxiv_results)


        return {"documents": documents,'question':question}
    
    
    def generate_answer(self,state):
        """
        Generate an answer to the user's question using retrieved documents.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            dict: Updated state with the generated answer.
        """

        question = state['question']
        documents = state['documents']

        self.logger.info("Generating Answer")
        generation = self.qa_rag_chain.invoke({'context':documents,
                                              'question':question})
        return {'documents':documents,'question':question,'generation':generation}
    
    def decide_to_generate(self,state):
        """
        Decide whether to generate an answer or rewrite the query.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            str: The next action ('rewrite_query' or 'generate_answer').
        """

        web_search_needed = state['web_search_needed']
        if web_search_needed=='Yes':
            self.logger.info("Some or all documents are irrelevant")
            return "rewrite_query"
        else:
            self.logger.info("all documents are relevant, generating answer")
            return 'generate_answer'
        
    def llm_router(self):
        """
        Create an LLM-based router to decide the appropriate tool or data source.

        Returns:
            Callable: A router function to decide between Wikipedia, web, or arXiv search.
        """

        llm = ChatGroq(model='Gemma2-9b-It')
        self.structured_llm_router = llm.with_structured_output(self.RouteQuery)
        

        system = """
                you are an expert at routing a user question to a arxiv or wikipedia or web.
                the vectorstore contains documents related to agents,prompt engineering and adverserial attacks.
                use the vectorstore for questions on these topics. Otherwise use wiki search
                """
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ('system',system),
                ("human","{question}"),
            ]
        )

        question_router = route_prompt|self.structured_llm_router
        self.logger.info("Created llm router to route to wikipedia, arxiv or web search")
        return question_router
    
    def route_question(self,state):
        """
        Route the user's question to the appropriate tool or data source.

        Args:
            state (GraphState): Current state of the graph.

        Returns:
            str: The next action ('wiki_search', 'arxiv_search', or 'web_search').
        """

        question_router = self.llm_router()
        question = state['question']
        source = question_router.invoke({'question':question})

        if source.datasource=='wiki_search':
            self.logger.info("---Routing to WikiSearch---")
            return 'wiki_search'
        elif source.datasource=='arxiv_search':
            self.logger.info("---Routing to VectorStore---")
            return 'arxiv_search'
        else:
            self.logger.info("---Routing to web search---")
            return 'web_search'
    

    def build_graph(self):
        """
        Build a state machine graph representing the workflow for question answering.

        This function creates a directed state graph with various nodes representing 
        different stages of the workflow, such as document retrieval, grading, 
        rewriting queries, and generating answers. It also includes decision nodes 
        for routing queries to different data sources and deciding whether to 
        generate an answer or rewrite the query.

        The graph includes the following states:
        - START: The entry point of the workflow.
        - retrieve: Retrieves documents relevant to the query.
        - grade_documents: Grades the relevance of the retrieved documents.
        - decide_to_generate: Decides whether to rewrite the query or generate an answer.
        - rewrite_query: Rewrites the query for better search results.
        - route_question: Routes the question to the appropriate data source.
        - wiki_search: Searches Wikipedia for relevant documents.
        - arxiv_search: Searches arXiv for relevant documents.
        - web_search: Performs a web search for additional context.
        - generate_answer: Generates the final answer using the available documents.
        - END: The endpoint of the workflow, with the generated answer.

        Returns:
            StateGraph: A state graph representing the workflow.
        """
        agentic_rag = StateGraph(self.GraphState)
        
        # Define Nodes
        agentic_rag.add_node("retrieve", self.retrieve)
        agentic_rag.add_node("grade_documents", self.grade_documents)
        agentic_rag.add_node("rewrite_query", self.rewrite_query)
        agentic_rag.add_node("wiki_search", self.wiki_search)
        agentic_rag.add_node("arxiv_search",self.arxiv_search)
        agentic_rag.add_node("web_search", self.web_search)
        agentic_rag.add_node("generate_answer", self.generate_answer)

        # Build Graph
        agentic_rag.add_edge(START, "retrieve")
        agentic_rag.add_edge("retrieve", "grade_documents")
        agentic_rag.add_conditional_edges("grade_documents",
                                           self.decide_to_generate,
                                           {'rewrite_query': "rewrite_query",'generate_answer': "generate_answer"},
                                           )
        agentic_rag.add_conditional_edges("rewrite_query",
                                          self.route_question,
                                          {'wiki_search': "wiki_search",'arxiv_search': "arxiv_search",'web_search': "web_search"},
                                          )
        agentic_rag.add_edge("wiki_search",'generate_answer')
        agentic_rag.add_edge("arxiv_search",'generate_answer')
        agentic_rag.add_edge("web_search",'generate_answer')
        agentic_rag.add_edge("generate_answer", END)
        agentic_rag = agentic_rag.compile()
        return agentic_rag


    


    


