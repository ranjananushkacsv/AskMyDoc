import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Imports for the RAG pipeline ---
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Set the path to the FAISS vector database
FAISS_DB_PATH = "vectorstore/db_faiss"
# Set the name of the Ollama model used for embeddings (must match the one used to create the DB)
OLLAMA_EMBEDDINGS_MODEL = "deepseek-r1:1.5b"
# Set the name of the LLM for answering the questions
LLM_MODEL = "mistral-small-latest"

try:
    # Initialize the embeddings model first, as it's required to load the FAISS DB
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL)
    
    # Load the pre-built FAISS vector database from the local directory
    # We use 'allow_dangerous_deserialization=True' for this specific FAISS method
    db_faiss = FAISS.load_local(
        FAISS_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("FAISS vector database loaded successfully.")

except Exception as e:
    print(f"Error loading FAISS database from '{FAISS_DB_PATH}': {e}")
    # Exit the script if the database cannot be loaded
    exit()

# Setup LLM model
llm_model = ChatMistralAI(model=LLM_MODEL)

# Retrieve documents based on a query
def retrive_docs(query):
    """
    Retrieves relevant documents from the loaded FAISS database.
    """
    return db_faiss.similarity_search(query)

# Get context from retrieved documents
def get_context(documents):
    """
    Combines the content of a list of documents into a single string.
    """
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Answer the question using a RAG chain
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    """
    Constructs a prompt with context and gets a response from the LLM.
    """
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

# Example Usage
#question = "What are the grounds for divorce and why?" 
#retrived_docs = retrive_docs(question)
#print("\nAI Lawyer:", answer_query(documents=retrived_docs, model=llm_model, query=question))
