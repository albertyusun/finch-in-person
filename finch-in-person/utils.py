"""
Utility functions for the Demand Letter Evaluator
"""

import os
import logging
from pathlib import Path
import base64
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Union

from pdfminer.high_level import extract_text  # pip install pdfminer.six
from pypdf import PdfReader                  # pip install pypdf
import pytesseract                            # pip install pytesseract pdf2image pillow
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Robust text extraction that handles:
      • normal / text-based PDFs   (PyPDF2 → fast)
      • PDFs with tricky encodings (pdfminer.six)
      • image-only / scanned PDFs  (OCR fallback)
    """
    pdf_path = Path(pdf_path)

    # Fast path – PyPDF2 works on most text-based PDFs
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if text.strip():
            return text
    except Exception as e:
        logger.debug(f"PyPDF2 failed on {pdf_path}: {e}")

    # Second try – pdfminer.six is slower but more tolerant
    try:
        text = extract_text(str(pdf_path))
        if text.strip():
            return text
    except Exception as e:
        logger.debug(f"pdfminer.six failed on {pdf_path}: {e}")

    # Fallback – probably a scanned (image-only) PDF → OCR each page
    try:
        images = convert_from_path(str(pdf_path), dpi=300)
        ocr_text = []
        for idx, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                ocr_text.append(page_text)
        if ocr_text:
            return "\n".join(ocr_text)
    except Exception as e:
        logger.error(f"OCR failed on {pdf_path}: {e}")

    logger.warning(f"Could not extract text from {pdf_path}")
    return ""

def process_source_documents(client, source_files):
    """
    Process source documents to extract key facts using OpenAI.
    
    Args:
        client: OpenAI client
        source_files: List of paths to source document PDFs
        
    Returns:
        Dictionary of extracted facts
    """
    # This would be a complex function in practice
    # For each document, we'd extract the text and then use OpenAI to extract key facts
    
    all_facts = {}
    
    # Process each source document
    for doc_path in source_files:
        logger.info(f"Processing source document: {doc_path}")
        doc_name = doc_path.stem
        
        # Extract text from PDF (in a real implementation)
        doc_text = extract_text_from_pdf(doc_path)
        
        # Use OpenAI to extract key facts
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a legal assistant who extracts key facts from legal and medical documents for personal injury cases."},
                {"role": "user", "content": f"Extract the most important facts from this document that would be relevant for a demand letter. Focus on dates, injuries, treatments, and damages.\n\nDocument: {doc_name}\n\n{doc_text}"}
            ],
            temperature=0.2
        )
        
        # Extract the facts
        facts = response.choices[0].message.content
        all_facts[doc_name] = facts
        
        # Avoid rate limiting
        time.sleep(0.5)
    
    # Compile all facts into a final summary
    fact_text = "\n\n".join([f"## {doc_name}\n{facts}" for doc_name, facts in all_facts.items()])
    
    # Use OpenAI to create a consolidated fact summary
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant who summarizes and organizes case facts for personal injury demand letters."},
            {"role": "user", "content": f"Based on these extracted facts from multiple documents, create a consolidated and organized summary of the most important facts for this case. Organize by categories like incident details, injuries, treatment, damages, etc.\n\n{fact_text}"}
        ],
        temperature=0.2
    )
    
    # Return the consolidated facts
    return {
        "consolidated_summary": response.choices[0].message.content,
        "individual_documents": all_facts
    }

def calculate_weighted_score(scores):
    """
    Calculate the weighted score based on category scores.
    
    Args:
        scores: Dictionary of category scores
        
    Returns:
        Weighted total score
    """
    # Updated weights to match the new evaluation criteria
    weights = {
        "Structure and Organization": 0.10,
        "Factual Presentation": 0.10,
        "Medical Documentation": 0.15,
        "Damages Calculation": 0.25,
        "Precedent and Legal Authority": 0.10,
        "Legal Strategy": 0.05,
        "Persuasiveness": 0.10,
        "Settlement Justification": 0.10,
        "Source Document Representation": 0.05
    }
    
    # Map to handle variations in category naming
    category_map = {
        "structure and organization": "Structure and Organization",
        "factual presentation": "Factual Presentation",
        "medical documentation": "Medical Documentation",
        "damages calculation": "Damages Calculation",
        "precedent and legal authority": "Precedent and Legal Authority",
        "legal strategy": "Legal Strategy",
        "persuasiveness": "Persuasiveness",
        "settlement justification": "Settlement Justification",
        "source document representation": "Source Document Representation"
    }
    
    total_score = 0
    total_weight = 0
    
    # Log for debugging
    logger.debug(f"Raw scores: {scores}")
    
    for category_raw, score_data in scores.items():
        # Clean up category name (remove heading markings, trim, lowercase for matching)
        category_clean = category_raw.replace("###", "").strip().lower()
        
        # Map to standard category name if possible
        if category_clean in category_map:
            category = category_map[category_clean]
        else:
            # Try to find a match by partial string
            category = None
            for key, value in category_map.items():
                if key in category_clean:
                    category = value
                    break
        
        # If we found a matching category with a weight
        if category and category in weights and "score" in score_data:
            score = score_data["score"]
            weight = weights[category]
            
            logger.debug(f"Adding score for {category}: {score} × {weight}")
            total_score += score * weight
            total_weight += weight
    
    # Normalize if not all categories were scored
    if total_weight > 0:
        # Calculate final score - just return the weighted sum (out of 5)
        final_score = total_score
        logger.debug(f"Final weighted score: {final_score}")
        return final_score
    else:
        logger.warning("No valid categories found for scoring")
        return 0