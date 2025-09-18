import streamlit as st
import os
import tempfile
from pathlib import Path

# Import your existing RAG pipeline functions
from rag_pipeline import answer_query, llm_model
# Add to your existing imports in frontend.py
from summarizer import generate_ai_summary, get_summary_metrics
from drafting import drafting_interface

# Import vector database functions for dynamic processing
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="AskMyDoc",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more polished look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #f0f2f6;
    }

    .css-1d391kg, .css-1y4p8bb {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px #388E3C;
    }

    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 2px #388E3C;
        transform: translateY(2px);
    }

    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #ddd;
        padding: 10px;
    }
    
    [data-testid="stChatMessage"][role="user"] {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
    }

    [data-testid="stChatMessage"][role="AI Lawyer"] {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
    }

    h1, h2, h3 {
        color: #1f77b4;
    }
    
    .summary-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""

# Functions for dynamic PDF processing
@st.cache_resource
def get_embeddings_model():
    """Initialize and cache the embeddings model"""
    try:
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings model: {e}")
        return None

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and create vector database"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load the PDF
        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()
        
        if not documents:
            st.error("No content found in the uploaded PDF.")
            return None
        
        # Extract full text for summarization
        full_text = "\n\n".join([doc.page_content for doc in documents])
        st.session_state.full_text = full_text
        
        # Create text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        text_chunks = text_splitter.split_documents(documents)
        
        # Get embeddings model
        embeddings = get_embeddings_model()
        if not embeddings:
            return None
        
        # Create vector database
        vector_db = FAISS.from_documents(text_chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return vector_db, len(documents), len(text_chunks)
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def retrieve_docs_from_db(query, vector_db):
    """Retrieve documents from the provided vector database"""
    if vector_db is None:
        return []
    return vector_db.similarity_search(query)

def get_context_from_docs(documents):
    """Get context from retrieved documents"""
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>AskMyDoc</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Upload your document and ask questions about its content.</p>", unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2,tab3 = st.tabs(["📄 Document Q&A", "📋 AI Summarizer", "📝 Document Drafting"])

with tab1:
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Upload a Document")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            accept_multiple_files=False,
            help="Upload a PDF document to analyze its content.",
            key="qa_uploader"
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            file_name = uploaded_file.name
            
            # Check if this is a new file
            if st.session_state.current_file_name != file_name:
                st.session_state.current_file_name = file_name
                
                with st.spinner("Processing your document... This may take a moment."):
                    result = process_uploaded_pdf(uploaded_file)
                    
                    if result:
                        vector_db, num_pages, num_chunks = result
                        st.session_state.vector_db = vector_db
                        
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"📄 Pages: {num_pages} | 📝 Text chunks: {num_chunks}")
                        
                        # Clear previous messages when new file is uploaded
                        st.session_state.messages = []
                    else:
                        st.error("❌ Failed to process the document. Please try again.")
            else:
                st.success(f"✅ Document '{file_name}' is ready for questions!")
        
        # Instructions
        if uploaded_file is None:
            st.info("👆 Please upload a PDF document to get started.")
        else:
            st.success("🎉 Document loaded! You can now ask questions in the chat.")

    with col2:
        st.subheader("2. Ask about your Document to our AI")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_query = st.chat_input("Ask a question about your document...", key="qa_chat")
        
        # Process user query
        if user_query:
            if uploaded_file is None:
                st.error("Please upload a PDF file first.")
            elif st.session_state.vector_db is None:
                st.error("Document is still being processed. Please wait.")
            else:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                # Generate response
                try:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Retrieve relevant documents
                            retrieved_docs = retrieve_docs_from_db(user_query, st.session_state.vector_db)
                            
                            if not retrieved_docs:
                                response = "I couldn't find relevant information in the document to answer your question. Please try rephrasing or ask about different content."
                            else:
                                # Generate answer using your existing function
                                response = answer_query(
                                    documents=retrieved_docs, 
                                    model=llm_model, 
                                    query=user_query
                                )
                            
                            st.write(response)
                            
                            # Add AI response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.error("Please ensure your models are running correctly.")

with tab2:
    st.subheader("📋 AI Document Summarizer")
    st.markdown("Generate an AI-powered summary of your uploaded document.")
    
    if st.session_state.full_text:
        original_word_count = len(st.session_state.full_text.split())
        st.info(f"📊 Loaded document contains approximately {original_word_count} words.")
        
        # Summary configuration
        col1, col2 = st.columns(2)
        with col1:
            summary_length = st.slider(
                "Summary Length (words)",
                min_value=100,
                max_value=500,
                value=250,
                help="Target length for the AI-generated summary"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")
            if st.button("🚀 Generate AI Summary", type="primary", use_container_width=True):
                if st.session_state.full_text:
                    with st.spinner("🤖 AI is analyzing and summarizing your document..."):
                        summary = generate_ai_summary(
                            st.session_state.full_text, 
                            summary_length=summary_length
                        )
                        st.session_state.summary = summary
                else:
                    st.warning("Please upload a document in the 'Document Q&A' tab first.")

        if st.session_state.summary:
            metrics = get_summary_metrics(st.session_state.full_text, st.session_state.summary)
            
            st.markdown(f"### 📝 AI-Generated Summary ({metrics['summary_words']} words)")
            st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
            st.markdown(st.session_state.summary)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Content Reduced", f"{metrics['reduction_percent']}%")
            with col2:
                st.metric("Original Words", metrics['original_words'])
            with col3:
                st.metric("Summary Words", metrics['summary_words'])
            
            # Download button for summary
            st.download_button(
                label="📥 Download Summary",
                data=st.session_state.summary,
                file_name="ai_document_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("📝 Please upload a PDF document in the 'Document Q&A' tab to enable summarization.")
        
        # Alternative: Allow direct text input for summarization
        st.subheader("Or summarize custom text:")
        custom_text = st.text_area(
            "Paste text to summarize",
            height=200,
            placeholder="Paste any text here for AI summarization...",
            key="custom_text"
        )
        
        if custom_text:
            custom_length = st.slider(
                "Summary Length",
                min_value=100,
                max_value=500,
                value=200,
                key="custom_length"
            )
            
            if st.button("🚀 Summarize Custom Text", key="custom_summarize"):
                with st.spinner("🤖 AI is analyzing and summarizing..."):
                    summary = generate_ai_summary(custom_text, custom_length)
                    st.markdown("### 📝 AI-Generated Summary")
                    st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                    st.markdown(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
with tab3:
    drafting_interface()
# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>AI Legal Assistant - Powered by RAG Technology & AI Summarization</p>", unsafe_allow_html=True)
