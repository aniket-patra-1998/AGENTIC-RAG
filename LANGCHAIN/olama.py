import os
from dotenv import load_dotenv
import bs4
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an assistanr. Answer the questions asked "),
        ('user','Question:{question}')
    ]
)

st.title('langchain demo with gemma:2b')
input_text = st.text_input("what do you have in mind?")

llm = Ollama(model = "gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))