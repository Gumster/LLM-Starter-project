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
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        # Persistent FAISS index directory
        self.index_dir = os.environ.get("VECTOR_INDEX_DIR", "faiss_index")
        # Attempt to load a persisted FAISS index on startup
        try:
            index_faiss = os.path.join(self.index_dir, "index.faiss")
            index_pkl = os.path.join(self.index_dir, "index.pkl")
            if os.path.exists(index_faiss) and os.path.exists(index_pkl):
                self.vector_store = FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded persisted FAISS index from {self.index_dir}")
        except Exception as e:
            print(f"Failed to load persisted FAISS index: {e}")

    def _ensure_vector_store(self) -> None:
        """Ensure the vector store is available by loading persisted or processing documents."""
        if not self.vector_store:
            try:
                index_faiss = os.path.join(self.index_dir, "index.faiss")
                index_pkl = os.path.join(self.index_dir, "index.pkl")
                if os.path.exists(index_faiss) and os.path.exists(index_pkl):
                    self.vector_store = FAISS.load_local(
                        self.index_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
            except Exception as e:
                print(f"Failed to load persisted FAISS index: {e}")
        if not self.vector_store:
            print("No vector store available. Processing documents first.")
            self.process_documents()
    
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
        
        # Persist vector store to disk
        try:
            self.vector_store.save_local(self.index_dir)
            print(f"Saved FAISS index to {self.index_dir}")
        except Exception as e:
            print(f"Failed to save FAISS index: {e}")
        
        print(f"Processed {len(documents)} documents into {len(splits)} chunks")
    
    def get_retriever(self, k: int = 8):
        """Return an MMR-based retriever configured for top-k retrieval."""
        self._ensure_vector_store()
        if not self.vector_store:
            return None
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(2 * k, 32), "lambda_mult": 0.5}
        )

    def get_relevant_documents(self, query: str, k: int = 8) -> List:
        """Get relevant documents for a query using MMR retrieval."""
        self._ensure_vector_store()
        if not self.vector_store:
            return []
        retriever = self.get_retriever(k=k)
        return retriever.get_relevant_documents(query) if retriever else []
    
    def rerank_documents(self, query: str, docs: List, top_k: int = 5) -> List:
        """Optionally rerank documents using a cross-encoder for better top-1 precision.
        Falls back to truncation if reranking model is unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            cross = CrossEncoder(model_name)
            pairs = [[query, getattr(d, 'page_content', '')] for d in docs]
            scores = cross.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked[:max(1, min(top_k, len(docs)))] ]
        except Exception as e:
            print(f"Rerank unavailable or failed ({e}). Using top-{top_k} without reranking.")
            return docs[:max(1, min(top_k, len(docs)))]
    
    def get_all_documents(self) -> List:
        """Get all loaded documents
        
        Returns:
            List of all documents
        """
        if not self.documents:
            self.load_documents()
        
        return self.documents