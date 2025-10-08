import os
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class DocumentProcessor:
    """Class to handle document loading, processing, and retrieval"""
    
    def __init__(self, documents_dir: str):
        """Initialize the document processor
        
        Args:
            documents_dir: Directory containing the documents
        """
        self.documents_dir = documents_dir
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def load_documents(self) -> List:
        """Load documents from the documents directory"""
        loaders = []
        
        # Text files loader
        text_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        loaders.append(text_loader)
        
        # PDF files loader
        pdf_loader = DirectoryLoader(
            self.documents_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        loaders.append(pdf_loader)
        
        # Load all documents
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading documents: {e}")
        
        self.documents = documents
        return documents
    
    def process_documents(self) -> None:
        """Process documents and create vector store"""
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("No documents found to process")
            return
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        print(f"Processed {len(documents)} documents into {len(splits)} chunks")
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List:
        """Get relevant documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            print("No vector store available. Processing documents first.")
            self.process_documents()
            
            if not self.vector_store:
                return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def get_all_documents(self) -> List:
        """Get all loaded documents
        
        Returns:
            List of all documents
        """
        if not self.documents:
            self.load_documents()
        
        return self.documents