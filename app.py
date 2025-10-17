import os
import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import shutil

from document_processor import DocumentProcessor
from llm_handler import LLMHandler

app = FastAPI()

# Create directories if they don't exist
os.makedirs("documents", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize document processor and LLM handler
document_processor = DocumentProcessor("documents")
llm_handler = LLMHandler()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to the documents folder"""
    uploaded_files = []
    for file in files:
        file_path = os.path.join("documents", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_files.append(file.filename)
    
    # Process the documents after upload
    document_processor.process_documents()
    
    return {"message": f"Successfully uploaded {len(uploaded_files)} files", "files": uploaded_files}

@app.post("/process")
async def process_documents():
    """Process all documents in the documents folder"""
    print("Processing documents...")
    document_processor.process_documents()
    print("Documents processed successfully")
    return {"message": "Documents processed successfully"}

@app.post("/query")
async def query_llm(prompt: str = Form(...), k: int = 8, rerank: bool = False):
    """Query the LLM with the given prompt using MMR retrieval and optional reranking.
    - k: number of documents to retrieve (query parameter)
    - rerank: whether to apply cross-encoder reranking (query parameter)
    """
    retriever = document_processor.get_retriever(k=k)
    docs = []
    if retriever:
        docs = retriever.get_relevant_documents(prompt)
        print(f"Retrieved {len(docs)} documents for query with k={k}")
        # Optional reranking for stronger top-1 accuracy
        if rerank and docs:
            try:
                docs = document_processor.rerank_documents(prompt, docs, top_k=min(k, len(docs)))
                print(f"Applied reranking; top-{len(docs)} docs reordered")
                # Build a temporary retriever from reranked docs to feed the QA chain
                from langchain.vectorstores import FAISS
                tmp_store = FAISS.from_documents(docs, document_processor.embeddings)
                retriever_to_use = tmp_store.as_retriever(search_kwargs={"k": len(docs)})
            except Exception as e:
                print(f"Reranking step failed ({e}); using original retriever")
                retriever_to_use = retriever
        else:
            retriever_to_use = retriever
        # Generate response from LLM using RetrievalQA chain
        response = llm_handler.generate_response_with_retriever(prompt, retriever_to_use)
    else:
        print("No retriever available. Falling back to context-less generation.")
        response = llm_handler.generate_response(prompt, [])
    
    return {"response": response, "sources": [doc.metadata.get("source", "Unknown") for doc in docs]}

@app.post("/train")
async def train_model():
    """Train/fine-tune the model on the documents"""
    print("Training model on documents...")
    result = llm_handler.train_on_documents(document_processor.get_all_documents())
    print("Model training completed")
    return {"message": result}

@app.get("/documents")
async def list_documents():
    """List all documents in the documents folder"""
    if not os.path.exists("documents"):
        return {"files": []}
    files = [f for f in os.listdir("documents") if os.path.isfile(os.path.join("documents", f))]
    return {"files": files}

@app.post("/reset")
async def reset_application():
    """Reset the application by removing all documents and resetting the vector store"""
    # Remove all documents
    if os.path.exists("documents"):
        for file in os.listdir("documents"):
            file_path = os.path.join("documents", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    # Remove persisted FAISS index directory
    index_dir = os.environ.get("VECTOR_INDEX_DIR", "faiss_index")
    try:
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            print(f"Removed FAISS index directory: {index_dir}")
    except Exception as e:
        print(f"Failed to remove FAISS index directory {index_dir}: {e}")
    
    # Reset the document processor (recreate the vector store)
    global document_processor
    document_processor = DocumentProcessor("documents")
    
    print("Application reset: All documents removed and vector store reset")
    return {"message": "Application reset successfully"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)