import streamlit as st
import os
import tempfile
from pathlib import Path

# Import your existing RAG pipeline functions
from rag_pipeline import answer_query, llm_model
# Add to your existing imports in frontend.py
from summarizer import generate_ai_summary, get_summary_metrics
from drafting import drafting_interface

# Import vector database functions for dynamic processing - FAISS ONLY
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  # ✅ FAISS only

st.set_page_config(
    page_title="AskMyDoc - PDF Analysis & Q&A",
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

    [data-testid="stChatMessage"][role="AI Assistant"] {
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
    
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
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
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Functions for dynamic PDF processing
@st.cache_resource
def get_embeddings_model():
    """Initialize and cache the embeddings model"""
    try:
        # Keep using nomic-embed-text (same as your rag_pipeline.py)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return embeddings
    except Exception as e:
        st.error(f"❌ Error initializing embeddings model: {e}")
        st.error("Please run: `ollama pull nomic-embed-text`")
        return None

# In the process_uploaded_pdf function, update the embeddings section:
# In the process_uploaded_pdf function, update the embeddings section:
def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and create FAISS vector database"""
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
        
        # Get embeddings model - UPDATED to use Ollama
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        if not embeddings:
            return None
        
        # ✅ USE FAISS
        vector_db = FAISS.from_documents(text_chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return vector_db, len(documents), len(text_chunks)
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return None

def retrieve_docs_from_db(query, vector_db):
    """Retrieve documents from the provided vector database"""
    if vector_db is None:
        return []
    try:
        return vector_db.similarity_search(query, k=3)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>AskMyDoc - PDF Analysis & Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Upload your PDF documents and get AI-powered analysis, summaries, and answers.</p>", unsafe_allow_html=True)

# Model status check
try:
    if llm_model is None:
        st.error("❌ AI model is not available. Please check if Ollama is running and models are installed.")
    else:
        st.success("✅ AI model is ready!")
except Exception as e:
    st.error(f"❌ Error loading AI model: {e}")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📄 Document Q&A", "📋 AI Summarizer", "📝 Document Drafting"])

with tab1:
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Upload a PDF Document")
        
        st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            accept_multiple_files=False,
            help="Upload a PDF document to analyze its content.",
            key="qa_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        if uploaded_file is not None:
            file_name = uploaded_file.name
            
            # Check if this is a new file
            if st.session_state.current_file_name != file_name:
                st.session_state.current_file_name = file_name
                st.session_state.processing_complete = False
                
                with st.spinner("🔄 Processing your document... This may take a moment."):
                    result = process_uploaded_pdf(uploaded_file)
                    
                    if result:
                        vector_db, num_pages, num_chunks = result
                        st.session_state.vector_db = vector_db
                        st.session_state.processing_complete = True
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Document processed successfully!")
                        st.info(f"📄 Pages: {num_pages} | 📝 Text chunks: {num_chunks}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Clear previous messages when new file is uploaded
                        st.session_state.messages = []
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error("❌ Failed to process the document. Please try again.")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                if st.session_state.processing_complete:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Document '{file_name}' is ready for questions!")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Instructions
        if uploaded_file is None:
            st.info("👆 Please upload a PDF document to get started.")
            st.markdown("""
            **Supported Features:**
            - 📖 PDF text extraction
            - 🤖 AI-powered Q&A
            - 🔍 Semantic search
            - 📊 Document analysis
            """)
        else:
            st.markdown("""
            **🎉 Document loaded! You can now:**
            - Ask questions about the content
            - Get AI-powered answers
            - Analyze document sections
            """)

    with col2:
        st.subheader("2. Ask Questions about Your Document")
        
        if uploaded_file is None:
            st.info("📝 Please upload a PDF document first to start asking questions.")
        elif not st.session_state.processing_complete:
            st.warning("⏳ Document is still being processed. Please wait...")
        else:
            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            user_query = st.chat_input("Ask a question about your document...", key="qa_chat")
            
            # Process user query
            if user_query:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                # Generate response
                try:
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Thinking..."):
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
                            
                            # Show source information
                            if retrieved_docs:
                                with st.expander("📚 Source Information"):
                                    st.write(f"Found {len(retrieved_docs)} relevant sections")
                                    for i, doc in enumerate(retrieved_docs[:2], 1):
                                        st.write(f"**Section {i}:** {doc.page_content[:200]}...")
                            
                            # Add AI response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"❌ An error occurred while generating the response: {e}")
                    st.error("Please ensure your AI models are running correctly.")

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
                        try:
                            summary = generate_ai_summary(
                                st.session_state.full_text, 
                                summary_length=summary_length
                            )
                            st.session_state.summary = summary
                            st.success("✅ Summary generated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {e}")
                else:
                    st.warning("⚠️ Please upload a document in the 'Document Q&A' tab first.")

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
                    try:
                        summary = generate_ai_summary(custom_text, custom_length)
                        st.markdown("### 📝 AI-Generated Summary")
                        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                        st.markdown(summary)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Download for custom text
                        st.download_button(
                            label="📥 Download Custom Summary",
                            data=summary,
                            file_name="custom_text_summary.txt",
                            mime="text/plain",
                            use_container_width=True,
                            key="custom_download"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {e}")

with tab3:
    drafting_interface()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>PDF Analysis Tool - Powered by FAISS & AI Technology</p>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **AskMyDoc** is an AI-powered PDF analysis tool that allows you to:
    
    - **Upload PDF documents**
    - **Ask questions** about content
    - **Generate AI summaries**
    - **Create new documents**
    
    **Requirements:**
    - Ollama running locally
    - Embedding models installed
    - LLM models installed
    
    **Quick Setup:**
    ```bash
    ollama pull nomic-embed-text
    ollama pull llama2:3b
    ```
    """)
    
    # Model status
    st.header("🔧 System Status")
    try:
        if llm_model is not None:
            st.success("✅ AI Model: Ready")
        else:
            st.error("❌ AI Model: Not Available")
    except:
        st.error("❌ AI Model: Not Available")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Debug information
    if st.checkbox("Show debug info"):
        st.write("Session state keys:", list(st.session_state.keys()))
        if st.session_state.vector_db:
            st.success("Vector DB: Loaded")
        else:
            st.info("Vector DB: Not loaded")