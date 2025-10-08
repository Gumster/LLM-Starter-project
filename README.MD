# Local LLM Document Assistant

A web application that allows you to upload, process, and query documents using a local LLM (Large Language Model). This application uses FastAPI for the backend, LangChain for document processing, and a local GGUF model for generating responses.

## Features

- Upload text and PDF documents
- Process documents to create embeddings and a vector store
- Query the LLM with questions about your documents
- Train/fine-tune the model on your documents
- Reset functionality to clear all documents and start fresh

## Requirements

- Python 3.8+
- A GGUF format LLM model (e.g., llama-2-7b-chat.Q2_K.gguf)
- Sufficient RAM to run the LLM (8GB+ recommended)

## Installation

1. Clone this repository or download the source code
2. Create and activate a virtual environment:
```bash
# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have a GGUF model file in the `models` directory
   - The default configuration expects a model named `llama-2-7b-chat.Q2_K.gguf`
   - You can download GGUF models from [HuggingFace](https://huggingface.co/models?sort=trending&search=gguf)

## Running the Application

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate
```

2. Start the application:
```bash
python3 app.py
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. **Upload Documents**: Click "Choose Documents" to select files, then click "Upload Documents"
2. **Process Documents**: After uploading, click "Process Documents" to generate embeddings
3. **Ask Questions**: Type your question in the input field and click "Send"
4. **Train Model**: Click "Train Model on Documents" to improve responses based on your documents
5. **Reset Application**: Click "Reset Application" to remove all documents and reset the vector store

## Troubleshooting

- **Import Errors**: Ensure you're running the application within the activated virtual environment
- **Model Loading Errors**: Verify that your GGUF model is in the `models` directory
- **Memory Issues**: If you encounter memory errors, try using a smaller GGUF model

## Directory Structure

- `app.py`: Main application file
- `document_processor.py`: Handles document loading and processing
- `llm_handler.py`: Manages interactions with the LLM
- `documents/`: Directory for uploaded documents
- `models/`: Directory for LLM models
- `static/`: Contains CSS and JavaScript files
- `templates/`: Contains HTML templates