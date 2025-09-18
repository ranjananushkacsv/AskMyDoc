# drafting.py
import streamlit as st
import html
from fpdf import FPDF
import base64
from datetime import datetime
import tempfile
import os
from streamlit.components.v1 import html as st_html
import re

class DraftingPDF(FPDF):
    def __init__(self, font_family="Arial", font_size=12):
        super().__init__()
        self.font_family = font_family
        self.font_size = font_size
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        pass
        
    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(content, filename="document_draft"):
    """Create a PDF from the drafted content"""
    pdf = DraftingPDF("Arial", 12)
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, filename, 0, 1, 'C')
    pdf.ln(10)
    
    # Add content
    pdf.set_font("Arial", '', 12)
    
    # Process content with basic formatting
    lines = content.split('\n')
    for line in lines:
        if line.strip() == '':
            pdf.ln()
            continue
            
        # Simple formatting detection
        if line.strip().startswith('**') and line.strip().endswith('**'):
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 5, line.strip()[2:-2])
            pdf.set_font("Arial", '', 12)
        elif line.strip().startswith('*') and line.strip().endswith('*'):
            pdf.set_font("Arial", 'I', 12)
            pdf.multi_cell(0, 5, line.strip()[1:-1])
            pdf.set_font("Arial", '', 12)
        elif line.strip().startswith('_') and line.strip().endswith('_'):
            pdf.set_font("Arial", 'U', 12)
            pdf.multi_cell(0, 5, line.strip()[1:-1])
            pdf.set_font("Arial", '', 12)
        else:
            pdf.multi_cell(0, 5, line)
    
    # Return PDF as bytes instead of saving to file
    return pdf.output(dest='S').encode('latin1')

def insert_text_at_cursor(text_area_key, insert_text):
    """Helper function to insert text at cursor position"""
    if text_area_key in st.session_state:
        current_text = st.session_state[text_area_key]
        # For simplicity, just append the text
        st.session_state[text_area_key] = current_text + insert_text
    return st.session_state.get(text_area_key, "")

def drafting_interface():
    """Main drafting interface function"""
    st.header("📝 Document Drafting")
    st.markdown("Create and format your documents with enhanced editing features.")
    
    # Initialize session state
    if 'document_content' not in st.session_state:
        st.session_state.document_content = ""
    
    # Quick Insert Toolbar
    st.subheader("⚡ Quick Insert Tools")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📅 Insert Date", use_container_width=True):
            today = datetime.now().strftime("%B %d, %Y")
            st.session_state.document_content += f"\n\nDate: {today}\n"
            st.rerun()
    
    with col2:
        if st.button("✍️ Signature Line", use_container_width=True):
            signature_line = "\n\n_________________________\nSignature\n\nDate: _____________\n"
            st.session_state.document_content += signature_line
            st.rerun()
    
    with col3:
        if st.button("📋 Letter Header", use_container_width=True):
            header = "**[DOCUMENT TITLE]**\n\nTo: [Recipient]\nFrom: [Your Name]\nDate: [Date]\nSubject: [Subject]\n\n"
            st.session_state.document_content = header + st.session_state.document_content
            st.rerun()
    
    with col4:
        if st.button("📄 Paragraph Break", use_container_width=True):
            paragraph_break = "\n\n---\n\n"
            st.session_state.document_content += paragraph_break
            st.rerun()
    
    with col5:
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state.document_content = ""
            st.rerun()
    
    # Formatting Tips
    with st.expander("💡 Formatting Tips"):
        st.markdown("""
        **Text Formatting for PDF:**
        - `**Bold Text**` → **Bold Text**
        - `*Italic Text*` → *Italic Text*
        - `_Underlined Text_` → Underlined Text
        
        **Quick Tips:**
        - Press Ctrl+A to select all text
        - Use the quick insert buttons above to add common elements
        - The PDF will preserve your formatting
        """)
    
    # Main text editor
    st.subheader("📝 Document Content")
    document_content = st.text_area(
        "Write your document here:",
        value=st.session_state.document_content,
        height=400,
        key="content_editor",
        help="Use **bold**, *italic*, _underline_ for formatting. Use the quick insert buttons above for common elements."
    )
    
    # Update session state
    st.session_state.document_content = document_content
    
    # Document settings and actions
    st.subheader("📄 Export Document")
    
    doc_name = st.text_input("📋 Document Name", value="my_document")
    
    # Export actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Generate PDF", use_container_width=True, type="primary"):
            if document_content.strip():
                with st.spinner("Creating PDF..."):
                    try:
                        # Generate PDF as bytes
                        pdf_bytes = create_pdf(document_content, doc_name)
                        
                        st.success("✅ PDF generated successfully!")
                        
                        # Immediate download button
                        st.download_button(
                            label="📥 Download PDF Now",
                            data=pdf_bytes,
                            file_name=f"{doc_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="secondary"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error creating PDF: {str(e)}")
                        st.error("Please check your document content and try again.")
            else:
                st.warning("⚠️ Please add some content to your document before generating PDF.")
    
    with col2:
        # Word count and stats
        if document_content.strip():
            word_count = len(document_content.split())
            char_count = len(document_content)
            line_count = len(document_content.split('\n'))
            
            st.info(f"""
            📊 **Document Stats:**
            - Words: {word_count:,}
            - Characters: {char_count:,}
            - Lines: {line_count:,}
            """)
        else:
            st.info("📊 **Document Stats:**\nStart writing to see statistics")

# For testing the module independently
if __name__ == "__main__":
    st.set_page_config(page_title="Document Drafting", layout="wide")
    drafting_interface()