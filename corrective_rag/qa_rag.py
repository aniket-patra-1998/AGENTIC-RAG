from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import logging

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class RAG():
    """
    Class for creating a Retrieval-Augmented Generation (RAG) pipeline.
    It handles tasks such as generating question-answering chains, rephrasing
    questions, and formatting documents for context.

    Attributes:
        groq_llm (ChatGroq): Large language model for question-answering tasks.
        rewrite_llm (ChatGroq): Large language model for rephrasing questions.
        logger (Logger): Logger instance for debugging and information logs.
        prompt_template (ChatPromptTemplate): Template for generating QA prompts.
    """

    def __init__(self):
        self.groq_llm = ChatGroq(model_name='Gemma2-9b-It')
        self.rewrite_llm = ChatGroq(model_name='Gemma2-9b-It',temperature = 0)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.prompt_template = None

    def create_prompt(self):
        """
        Create a prompt template for the question-answering (QA) task.
        This prompt guides the LLM to use retrieved context for answering questions.
        """
        prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
        self.prompt_template = ChatPromptTemplate.from_template(prompt)
        

    def format_docs(self,docs):
        """
        Format a list of documents into a single string with each document's content separated by double newlines.

        Args:
            docs (list): List of documents to format.

        Returns:
            str: Formatted string containing document content.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_rag_chain(self):
        """
        Create a RAG chain using the created prompt template and LLM.
        The chain handles question-answering by utilizing retrieved documents as context.

        Returns:
            Chain or None: A configured RAG chain if successful, otherwise None.
        """
        self.create_prompt()
        if self.prompt_template is None:
            self.logger.error("Failed to create prompt template")
            return None
        else:
            self.logger.info("created a prompt template for qa rag")
            qa_rag_chain = (
                {
                    "context":(itemgetter('context')
                            |RunnableLambda(self.format_docs)),
                    "question":itemgetter('question')
                }
                |
                self.prompt_template
                |
                self.groq_llm
                |
                StrOutputParser()
            )
            self.logger.info("created a qa rag chain")
            return qa_rag_chain
        
    def rephraser(self):
        """
        Create a chain for rephrasing questions to optimize them for web search.
        The chain uses a prompt to guide the LLM in rephrasing the questions.

        Returns:
            Chain: A configured rephraser chain.
        """

        rephrase_prompt = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search by including official names whereever applicable.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
                 - Give me only one answer ans the best one
                 
             """
        
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', rephrase_prompt),
                ('human',"""Here is the initial question:
                            {question}
                            Formulate an improved question"""
                ),
            ]
        )
        
        question_rewriter = (
            rewrite_prompt|self.rewrite_llm|StrOutputParser()
        )
        self.logger.info("created a rephraser chain")
        return question_rewriter
        

