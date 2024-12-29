from retriever import Retriever
from query_grader import Grader
from qa_rag import RAG
from tools import Tools
from graphstate import Graph
from IPython.display import Image, display, Markdown



similarity_threshold_retriver = Retriever().create_retriever()
doc_grader = Grader().create_grader()
question_rewriter = RAG().rephraser()
web_tool = Tools().tavily_search_tool()
wiki_tool = Tools().wiki_search_tool()
arxiv_tool = Tools().arxiv_search_tool()
qa_rag_chain = RAG().create_rag_chain()

graph = Graph(similarity_threshold_retriever=similarity_threshold_retriver,
              doc_grader=doc_grader,
              question_rewriter=question_rewriter,
              web_search_tool=web_tool,
              wiki_search_tool=wiki_tool,
              arxiv_search_tool=arxiv_tool,
              qa_rag_chain=qa_rag_chain).build_graph()

query = input()

response = graph.invoke({"question": query})

display(Markdown(response['generation']))