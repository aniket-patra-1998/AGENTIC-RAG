import gzip
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import WebBaseLoader
import os
import logging

class Retriever():
    """
    A class to create a document retriever pipeline using Wikipedia or web-based data sources.
    This pipeline supports document loading, splitting, vector store creation, and similarity-based retrieval.

    Attributes:
        file_path (str): Path to the compressed JSONL file containing Wikipedia data.
        docs (list): List of loaded documents.
        doc_chunks (list or None): List of document chunks created by splitting loaded documents.
        chroma_db (Chroma or None): Placeholder for a Chroma database (not used in this implementation).
        similarity_retriver (Retriever or None): Similarity retriever for querying vectorized documents.
        take_web_data (bool): Flag to determine whether to load web data or Wikipedia data.
        logger (Logger): Logger instance for debugging and informational logs.
    """
    def __init__(self):
        self.file_path = 'simplewiki-2020-11-01.jsonl.gz'
        self.docs = []
        self.doc_chunks = None
        self.chroma_db = None
        self.similarity_retriver = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.take_web_data  = True


    def wiki_loader(self):
        """
        Loads documents from a compressed Wikipedia JSONL file.
        Filters the documents to include only those related to 'India' for faster computation.

        Raises:
            Exception: If there is an error during document loading.
        """
        #docs = []
        try:
            with gzip.open(self.file_path,'rt', encoding='utf8') as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    self.docs.append({
                        'metadata':{
                            'title':data.get('title'),
                            'article_id':data.get('id')
                        },
                        'data':" ".join(data.get("paragraphs")[0:3])
                        # restrict data to 3 paragraphs for simplicity and fast processing
                    })
            # Subset the data to use wikipedia data for india only for faster computation
            
            self.docs = [doc for doc in self.docs for x in ['india'] if x in doc['data'].lower().split()]
            

            self.docs = [Document(page_content=doc['data'],
                            metadata = doc['metadata']) for doc in self.docs]
            
            self.logger.info(f"Loaded {len(self.docs)} documents.")
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise
        
    def web_loader(self):
        """
        Loads documents from a specified web URL using a web-based loader.

        Raises:
            Exception: If there is an error during document loading.
        """
        try:
            loader = WebBaseLoader('https://www.webmd.com/arthritis/hip-replacement-surgery')
            self.docs = loader.load()

            
            self.logger.info(f"Loaded {len(self.docs)} documents.")
        except Exception as e:
            self.logger.error(f"Failed to load documents: {e}")
            raise


    def split_docs(self):
        """
        Splits loaded documents into smaller chunks using a recursive character-based splitter.

        Raises:
            Exception: If no documents are loaded.
        """
        if len(self.docs) > 0:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
            self.doc_chunks = splitter.split_documents(self.docs)
            
            self.logger.info(f"Split documents into {len(self.doc_chunks)} chunks.")
        else:
            self.logger.error("Failed to split documents as document is empty")

    def create_vectorstore(self):
        """
        Creates or loads a FAISS vector store for document similarity retrieval.
        Uses Ollama embeddings for vectorization.
        """

        ollama_model = OllamaEmbeddings()
        index_path = "./faiss_index"
        if os.path.exists(index_path):
            self.logger.info("FAISS index already exists. Loading existing index...")
            self.vectordb = FAISS.load_local(index_path,
                                             embeddings=ollama_model,
                                             allow_dangerous_deserialization=True)
            self.logger.info("Loaded existing FAISS index!")

        else:
            print("Creating new FAISS index...")
            
            self.vectordb = FAISS.from_documents(self.doc_chunks, ollama_model)
            self.vectordb.save_local(index_path)
            self.logger.info("FAISS index created and saved.")
        
        
    def get_retriver(self):
        """
        Configures a similarity-based retriever using the FAISS index.
        """

        # retriever using Chomadb
        """
        self.similarity_retriver = self.chroma_db.as_retriever(
            search_type = 'similarity_score_threshold',
            search_kwargs= {"k":3,
                            "score_threshold":0.3}
        ) 
        """

        # Retiever as FAISS index
        self.similarity_retriver= self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        self.logger.info("Retriever created successfully")
        
    
    def create_retriever(self):
        """
        Creates a complete retriever pipeline by loading data, splitting documents,
        creating a vector store, and configuring the similarity retriever.

        Returns:
            Retriever or None: The configured similarity retriever, or None if there is an error.

        Raises:
            ValueError: If no valid documents are found.
        """
        
        # load the docs
        try:
            if self.take_web_data:
                self.web_loader()
            else:
                self.wiki_loader()
            if len(self.docs) == 0:
                # create document chunks
                raise ValueError("No valid documents found.")
            self.split_docs()

            self.create_vectorstore()

            self.get_retriver()

            return self.similarity_retriver
        except Exception as e:
            self.logger.error(f"Error creating retriever: {e}")
            return None
        
    
                    

    
