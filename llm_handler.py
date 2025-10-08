import os
from typing import List, Dict, Any, Optional
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LLMHandler:
    """Class to handle interactions with the local LLM"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the LLM handler
        
        Args:
            model_path: Path to the local LLM model file
        """
        self.model_path = model_path or os.environ.get("LLM_MODEL_PATH", "models/llama-2-7b-chat.Q2_K.gguf")
        self.llm = None
        self.initialize_llm()
    
    def initialize_llm(self) -> None:
        """Initialize the LLM"""
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found at {self.model_path}. Please download a compatible model.")
                print("You can use models like Llama 2, Mistral, or other GGUF format models.")
                return
            
            # Set up callback manager for streaming output
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Initialize the LLM
            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=0.7,
                max_tokens=2000,
                top_p=1,
                callback_manager=callback_manager,
                verbose=True,
                n_ctx=4096  # Context window size
            )
            
            print(f"LLM initialized successfully using model: {self.model_path}")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
    
    def generate_response(self, prompt: str, relevant_docs: List = None) -> str:
        """Generate a response from the LLM for the given prompt
        
        Args:
            prompt: User prompt
            relevant_docs: List of relevant documents for context
            
        Returns:
            Generated response
        """
        if not self.llm:
            return "LLM not initialized. Please check if the model file exists and is accessible."
        
        try:
            # Create context from relevant documents
            context = ""
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt template with context
            template = """
            You are a helpful AI assistant. Use the following context to answer the question.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                context=context if context else "No relevant context found.",
                question=prompt
            )
            
            # Generate response
            response = self.llm(formatted_prompt)
            return response.strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def train_on_documents(self, documents: List) -> str:
        """Train/fine-tune the model on the provided documents
        
        Args:
            documents: List of documents to train on
            
        Returns:
            Status message
        """
        # Note: Full fine-tuning requires significant resources and specialized code
        # This is a simplified placeholder that would need to be expanded based on the specific LLM being used
        
        if not documents:
            return "No documents provided for training"
        
        try:
            # In a real implementation, this would involve:
            # 1. Preparing the training data
            # 2. Setting up the fine-tuning process
            # 3. Running the fine-tuning
            # 4. Loading the fine-tuned model
            
            return "Fine-tuning local LLMs requires specialized setup. Please refer to the model's documentation for specific fine-tuning instructions. Your documents have been processed and are ready for use with the retrieval system."
        
        except Exception as e:
            return f"Error during training: {str(e)}"