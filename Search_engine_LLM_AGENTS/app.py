import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import pickle
from langchain.tools.retriever import create_retriever_tool
#load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY")


## Arxiv and wikipedia tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrpper_axiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrpper_axiv)

search = DuckDuckGoSearchRun(name='Search')

with open(r'C:\Users\anike\Desktop\GEN-AI\TOOLS_AGENTS_LANGCHIN\retriever_arthoplasty.pkl', 'rb') as f: 
    retriever_arthoplasty = pickle.load(f)
retriver_tool = create_retriever_tool(retriever_arthoplasty,"langsmith-search","Search Any info about langsmith")


st.title("Langchain - Chat with Search")


## Side bar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role':"assistant",
         "content":"Hi, I am a chatbot who can search the web. How can I be of your help"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


if prompt:=st.chat_input(placeholder='What is machine learning?'):
    st.session_state.messages.append({'role':"user",
                                      'content': prompt})
    
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key,model_name = 'llama-3.1-70b-versatile',streaming=True)

    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)



