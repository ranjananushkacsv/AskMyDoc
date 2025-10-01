import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document

# --- Configuration ---
PDFS_DIRECTORY = 'pdfs'
FAISS_DB_PATH = "vectorstore/db_faiss"

def load_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF document from a given file path.
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
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks

def get_embeddings_model():
    """
    Initializes and returns an Ollama embeddings model.
    """
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        print("✅ Successfully initialized Ollama embeddings model")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        return None

def create_and_save_vector_db(text_chunks: List[Document], embeddings):
    """
    Creates a FAISS vector database from text chunks and saves it locally.
    """
    if not embeddings:
        print("Embeddings model not available. Cannot create vector database.")
        return

    try:
        print(f"Creating FAISS vector store and saving to {FAISS_DB_PATH}...")
        faiss_db = FAISS.from_documents(text_chunks, embeddings)
        faiss_db.save_local(FAISS_DB_PATH)
        print("FAISS vector database created and saved successfully.")
        return faiss_db
    except Exception as e:
        print(f"Error creating or saving FAISS database: {e}")
        return None

def main():
    """
    Main function to orchestrate the RAG setup process.
    """
    # Check if PDF directory exists
    if not os.path.exists(PDFS_DIRECTORY):
        os.makedirs(PDFS_DIRECTORY)
        print(f"Created '{PDFS_DIRECTORY}' directory. Please add your PDF files there.")
        return

    # Find PDF files
    pdf_files = [f for f in os.listdir(PDFS_DIRECTORY) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{PDFS_DIRECTORY}' directory.")
        print("Please add PDF files to the 'pdfs' directory and try again.")
        return

    all_documents = []
    
    # Process all PDF files
    for pdf_file in pdf_files:
        file_path = os.path.join(PDFS_DIRECTORY, pdf_file)
        print(f"Processing {pdf_file}...")
        
        documents = load_pdf(file_path)
        if documents:
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages from {pdf_file}")

    if not all_documents:
        print("No documents were loaded. Exiting.")
        return

    # Create text chunks from all documents
    print("Creating text chunks...")
    text_chunks = create_chunks(all_documents)

    # Initialize the embeddings model
    print("Initializing embeddings model...")
    embeddings_model = get_embeddings_model()
    if not embeddings_model:
        print("Exiting due to error in embedding model initialization.")
        return

    # Create and save the FAISS vector database
    create_and_save_vector_db(text_chunks, embeddings_model)

if __name__ == "__main__":
    main()