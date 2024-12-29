from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


class Grader():
    """
    A class for grading the relevance of retrieved documents to a user question.
    It uses a structured output model to provide binary grading results
    ('yes' or 'no') based on the relevance of document content.

    Attributes:
        llm (ChatGroq): Large language model used for grading tasks.
        doc_grader (Chain or None): Configured grading chain.
        logger (Logger): Logger instance for debugging and informational logs.
    """

    class GradeDocuments(BaseModel):
        """
        Schema for grading documents.
        Provides a binary score ('yes' or 'no') for the relevance of a document
        to a specific question.

        Attributes:
            binary_score (str): Indicates whether the document is relevant ('yes') or not ('no').
        """
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    def __init__(self):
        self.llm = ChatGroq(model_name='Gemma2-9b-It')
        self.doc_grader = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    def create_grader(self):
        """
        Create a grading chain for evaluating the relevance of retrieved documents.
        The chain uses a structured output model to provide binary scores.

        Returns:
            Chain: A configured chain for document grading.
        """

        structured_llm_grader = self.llm.with_structured_output(self.GradeDocuments)

        sys_prompt = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ('system',sys_prompt),
                ('human',""" Retrieved document:
                                {document}
                                User question: {question}"""),
            ]
        )
        self.doc_grader = (grade_prompt|structured_llm_grader)
        self.logger.info("Created a query grader successfully.")

        return self.doc_grader
         

