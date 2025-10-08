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
async def query_llm(prompt: str = Form(...)):
    """Query the LLM with the given prompt"""
    # Get relevant documents for the prompt
    relevant_docs = document_processor.get_relevant_documents(prompt)
    print("Processing query... "+prompt)
    # Generate response from LLM
    response = llm_handler.generate_response(prompt, relevant_docs)
    
    return {"response": response, "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs]}

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
    
    # Reset the document processor (recreate the vector store)
    global document_processor
    document_processor = DocumentProcessor("documents")
    
    print("Application reset: All documents removed and vector store reset")
    return {"message": "Application reset successfully"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)