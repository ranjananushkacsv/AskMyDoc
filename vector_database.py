import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document

# --- Configuration ---
# Set the directory for storing PDFs and the FAISS vector database.
PDFS_DIRECTORY = 'pdfs'
FAISS_DB_PATH = "vectorstore/db_faiss"

# Set the Ollama model name for embeddings.
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"


# --- File and Document Handling Functions ---
def load_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF document from a given file path.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of Document objects from the PDF content.
    """
    try:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        if not documents:
            print(f"Warning: No content found in {file_path}")
        return documents
    except Exception as e:
        print(f"Error loading PDF from {file_path}: {e}")
        return []

def create_chunks(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller text chunks for better retrieval.

    Args:
        documents: A list of Document objects.

    Returns:
        A list of text chunks (Document objects).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks


# --- Embeddings and Vector Store Functions ---
def get_embeddings_model(model_name: str) -> OllamaEmbeddings:
    """
    Initializes and returns an OllamaEmbeddings model.

    Args:
        model_name: The name of the Ollama model to use.

    Returns:
        An OllamaEmbeddings object.
    """
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        print(f"Successfully initialized Ollama embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        print(f"Error initializing Ollama embeddings model: {e}")
        return None

def create_and_save_vector_db(text_chunks: List[Document], embeddings: OllamaEmbeddings):
    """
    Creates a FAISS vector database from text chunks and saves it locally.

    Args:
        text_chunks: The list of Document chunks.
        embeddings: The OllamaEmbeddings model.
    """
    if not embeddings:
        print("Embeddings model not available. Cannot create vector database.")
        return

    try:
        print(f"Creating FAISS vector store and saving to {FAISS_DB_PATH}...")
        faiss_db = FAISS.from_documents(text_chunks, embeddings)
        faiss_db.save_local(FAISS_DB_PATH)
        print("FAISS vector database created and saved successfully.")
    except Exception as e:
        print(f"Error creating or saving FAISS database: {e}")


# --- Main Execution Block ---
def main():
    """
    Main function to orchestrate the RAG setup process.
    """
    # 1. Check if the PDF file exists before proceeding.
    file_path = os.path.join(PDFS_DIRECTORY, 'divorce.pdf')
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        print("Please place 'divorce.pdf' inside the 'pdfs' directory and try again.")
        return

    # 2. Load the PDF documents.
    print("Loading documents...")
    documents = load_pdf(file_path)
    if not documents:
        print("Exiting due to an error in document loading.")
        return
    print(f"Loaded {len(documents)} pages from the PDF.")

    # 3. Create text chunks from the documents.
    print("Creating text chunks...")
    text_chunks = create_chunks(documents)

    # 4. Initialize the embeddings model.
    print("Initializing embeddings model...")
    embeddings_model = get_embeddings_model(OLLAMA_MODEL_NAME)
    if not embeddings_model:
        print("Exiting due to an error in embedding model initialization.")
        return

    # 5. Create and save the FAISS vector database.
    create_and_save_vector_db(text_chunks, embeddings_model)

if __name__ == "__main__":
    main()
