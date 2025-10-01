import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Imports for the RAG pipeline ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Set the path to the FAISS vector database
FAISS_DB_PATH = "vectorstore/db_faiss"

try:
    # Initialize the embeddings model using Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("✅ Embeddings model loaded successfully")
    
    # Load the pre-built FAISS vector database from the local directory
    db_faiss = FAISS.load_local(
        FAISS_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("FAISS vector database loaded successfully.")

except Exception as e:
    print(f"Error loading FAISS database from '{FAISS_DB_PATH}': {e}")
    db_faiss = None

# Setup LLM model using Ollama (lighter than HuggingFace)
try:
    print("Loading LLM model: llama2")
    llm_model = Ollama(model="llama2")
    print("✅ Successfully loaded LLM model: llama2")
    
except Exception as e:
    print(f"❌ Error loading LLM model: {e}")
    print("Using a simple fallback model...")
    
    # Fallback: Use a very simple model
    class SimpleLLM:
        def invoke(self, prompt):
            if hasattr(prompt, 'content'):
                text = prompt.content
            elif isinstance(prompt, list) and len(prompt) > 0:
                text = str(prompt[-1]) if hasattr(prompt[-1], 'content') else str(prompt[-1])
            else:
                text = str(prompt)
            
            response = f"I've processed your query: '{text[:100]}...'. This is a fallback response since the main model couldn't load."
            return type('Response', (), {'content': response})()
    
    llm_model = SimpleLLM()

# Retrieve documents based on a query
def retrive_docs(query):
    """
    Retrieves relevant documents from the loaded FAISS database.
    """
    if db_faiss is None:
        return []
    try:
        return db_faiss.similarity_search(query, k=3)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

# Get context from retrieved documents
def get_context(documents):
    """
    Combines the content of a list of documents into a single string.
    """
    if not documents:
        return ""
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Custom prompt template
custom_prompt_template = """
You are a helpful AI assistant for PDF document analysis. Use the information from the document context to answer the user's question.

Guidelines:
- Answer based only on the provided context
- Be clear and concise
- If the answer isn't in the context, say so
- Format your response in a readable way

Question: {question}
Context: {context}

Answer:
"""

def answer_query(documents, model, query):
    """
    Constructs a prompt with context and gets a response from the LLM.
    """
    if not documents:
        return "I couldn't find relevant information in the document to answer this question. Please try rephrasing or ask about different content."
    
    if model is None:
        return "AI model is not available. Please check the model configuration."
    
    try:
        context = get_context(documents)
        prompt = custom_prompt_template.format(question=query, context=context)
        
        # Get response from the model
        response = model.invoke(prompt)
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"