# summarizer.py
from rag_pipeline import llm_model  # Import your existing LLM model

def generate_ai_summary(text, summary_length=250):
    """
    Generate a proper abstractive summary using the Qwen model
    """
    try:
        # Create a smart prompt for document summarization
        prompt_text = f"""You are an expert assistant. Create a concise, coherent summary of the provided document.

GUIDELINES:
- Understand the core concepts and main points
- Generate NEW sentences that capture the essence (do not copy paste text)
- Maintain accuracy and terminology
- Focus on key information and important details
- Keep the summary clear and professional
- Target length: {summary_length} words
- Write in complete, flowing paragraphs

DOCUMENT TEXT:

{text[:10000]}  # Truncate very long documents

Please provide a comprehensive summary:"""
        
        # Generate the summary using your Qwen model
        response = llm_model.invoke(prompt_text)
        
        return response.strip()
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_summary_metrics(original_text, summary_text):
    """
    Calculate summary metrics
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