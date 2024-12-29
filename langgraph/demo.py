import os
from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Course Langgraph"

from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langgraph.prebuilt import ToolNode,tools_condition

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [wiki_tool,arxiv_tool]

from langchain_groq import ChatGroq
llm = ChatGroq(model_name='Gemma2-9b-It')
llm = llm.bind_tools(tools=tools)

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State) 

def chatbot(state:State):
    return {"messages":llm.invoke(state['messages'])}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,'chatbot')
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools",tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge('chatbot',END)

graph = graph_builder.compile()




while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit','q']:
        print("Good Bye")
        break
    for event in graph.stream({'messages':("user",user_input)}):
        print(event.values())
        for value in event.values():
            #print(value['messages'])
            print("Assistant:",value['messages'])
