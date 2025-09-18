# summarizer.py
from langchain_core.prompts import ChatPromptTemplate
from rag_pipeline import llm_model  # Import your existing LLM model

def generate_ai_summary(text, summary_length=250):
    """
    Generate a proper abstractive summary using the LLM
    
    Args:
        text (str): The text content to summarize
        summary_length (int): Target word count for the summary
        
    Returns:
        str: AI-generated summary
    """
    try:
        # Create a smart prompt for legal document summarization
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert legal assistant. Create a concise, coherent summary of the provided legal document.
            
            GUIDELINES:
            - Understand the core concepts and main points
            - Generate NEW sentences that capture the essence (do not copy paste text)
            - Maintain legal accuracy and terminology
            - Focus on key clauses, obligations, rights, and important details
            - Keep the summary clear and professional
            - Target length: {summary_length} words
            - Write in complete, flowing paragraphs"""),
            ("human", "DOCUMENT TEXT:\n\n{text}\n\nPlease provide a comprehensive summary:")
        ])
        
        # Truncate very long documents to avoid context limits while keeping important content
        # Keep the beginning and end which often contains key information
        if len(text) > 12000:
            half_limit = 6000
            truncated_text = text[:half_limit] + "\n\n[...document continues...]\n\n" + text[-half_limit:]
        else:
            truncated_text = text
        
        # Format the prompt
        prompt = prompt_template.format_messages(
            summary_length=summary_length,
            text=truncated_text
        )
        
        # Generate the summary using your existing LLM
        response = llm_model.invoke(prompt)
        
        return response.content.strip()
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_summary_metrics(original_text, summary_text):
    """
    Calculate summary metrics
    
    Args:
        original_text (str): Original document text
        summary_text (str): Generated summary text
        
    Returns:
        dict: Summary metrics
    """
    original_words = len(original_text.split())
    summary_words = len(summary_text.split())
    
    if original_words > 0:
        reduction_percent = ((original_words - summary_words) / original_words) * 100
    else:
        reduction_percent = 0
    
    return {
        "original_words": original_words,
        "summary_words": summary_words,
        "reduction_percent": round(reduction_percent, 1)
    }

# Optional: If you want to test the summarizer independently
if __name__ == "__main__":
    # Test with sample text
    test_text = """
    This Agreement is made and entered into as of [Date], by and between [Company Name], 
    a corporation organized and existing under the laws of [State], with its principal 
    office located at [Address] ("Company"), and [Employee Name], an individual residing 
    at [Address] ("Employee").
    
    WHEREAS, Company desires to employ Employee, and Employee desires to be employed by 
    Company, subject to the terms and conditions set forth herein;
    
    NOW, THEREFORE, in consideration of the mutual covenants and promises contained herein, 
    the parties agree as follows:
    
    1. Employment. Company hereby employs Employee as [Position], and Employee accepts 
    such employment, subject to the terms and conditions of this Agreement.
    
    2. Term. The term of this Agreement shall commence on [Start Date] and shall continue 
    until terminated by either party in accordance with Section 5 hereof.
    
    3. Compensation. As compensation for services rendered hereunder, Company shall pay 
    Employee a base salary of $[Amount] per year, payable in accordance with Company's 
    standard payroll practices.
    
    4. Duties. Employee shall perform such duties as are customarily performed by one 
    holding such position and such other duties as may be assigned by Company from time to time.
    
    5. Termination. This Agreement may be terminated by either party upon [Number] days 
    written notice to the other party.
    """
    
    print("Testing AI Summarizer...")
    summary = generate_ai_summary(test_text, 150)
    print("\n" + "="*50)
    print("AI SUMMARY:")
    print("="*50)
    print(summary)
    
    metrics = get_summary_metrics(test_text, summary)
    print(f"\nOriginal: {metrics['original_words']} words")
    print(f"Summary: {metrics['summary_words']} words")
    print(f"Reduction: {metrics['reduction_percent']}%")