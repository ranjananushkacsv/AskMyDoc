# 📄 AskMyDoc - AI Document Analysis & Q&A System

AskMyDoc is an intelligent document analysis platform that leverages AI to help you understand, summarize, and interact with your PDF documents through natural language conversations.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-00A98F?style=for-the-badge)

## 🚀 Features

### 🤖 AI-Powered Document Q&A
- Upload PDF documents and ask questions about their content
- Get accurate, context-aware answers using Retrieval-Augmented Generation (RAG)
- Semantic search powered by FAISS vector database

### 📋 Smart Document Summarization
- Generate concise AI summaries of lengthy documents
- Customizable summary length (100-500 words)
- Content reduction metrics and analysis
- Support for both uploaded documents and custom text input

### ✍️ Intelligent Document Drafting
- Rich text editor with formatting support
- Quick insert tools for dates, signatures, and headers
- Export to professionally formatted PDF
- Document statistics and word count tracking

## 🛠️ Technical Architecture

### Core Components
- **Frontend**: Streamlit web application
- **AI Backend**: Ollama with Llama2 LLM(Experimenting with other models as well)
- **Vector Database**: FAISS for efficient similarity search
- **Embeddings**: Nomic-embed-text model
- **PDF Processing**: PDFPlumber for text extraction

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Ollama installed locally
- Git

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd AskMyDoc
```
### Step 1: Clone the Repository
```bash
pip install -r requirements.txt
```
### Step 3: Install Ollama & Download Models
```bash
# Install Ollama (visit https://ollama.ai for detailed instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required AI models
ollama pull nomic-embed-text
ollama pull llama2

