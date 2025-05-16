#!/usr/bin/env python3
"""
Demand Letter Evaluator - Main Script
Evaluates legal demand letters against source documents using OpenAI GPT models.
"""

import os
import json
import argparse
from pathlib import Path
import logging
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils import extract_text_from_pdf, process_source_documents, calculate_weighted_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()

def setup_folders():
    """Create necessary folders if they don't exist."""
    folders = [
        "data/source_documents",
        "data/demand_letters",
        "data/extracted_facts",
        "data/results"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        
    logger.info("Folder structure verified.")

def extract_facts_from_source_documents(force_reprocess=False):
    """
    Process source documents to extract key facts.
    
    Args:
        force_reprocess: If True, reprocess documents even if facts already exist
    
    Returns:
        Dictionary of extracted facts
    """
    facts_file = Path("data/extracted_facts/case_facts.json")
    
    # Check if facts.json already exists and can be loaded directly
    if facts_file.exists() and not force_reprocess:
        logger.info("Loading existing extracted facts.")
        try:
            with open(facts_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing facts file: {e}")
            # Continue to reprocess if loading fails
    
    # Handle the case where we have a JSON file directly
    if Path("data/source_documents/facts.json").exists():
        logger.info("Found facts.json in source_documents, using it directly.")
        try:
            with open(Path("data/source_documents/facts.json"), 'r') as f:
                facts = json.load(f)
            
            # Save to extracted_facts location
            with open(facts_file, 'w') as f:
                json.dump(facts, f, indent=2)
            
            return facts
        except Exception as e:
            logger.error(f"Error loading facts.json from source_documents: {e}")
            # Continue to standard processing if this fails
    
    # Check if we have a plain text file with facts
    text_facts = list(Path("data/source_documents").glob("*.txt"))
    if text_facts:
        logger.info(f"Found {len(text_facts)} text files in source_documents, processing them.")
        facts = {"consolidated_summary": "", "individual_documents": {}}
        
        # Process each text file
        for txt_file in text_facts:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if this is a JSON file
                if txt_file.stem.lower().endswith('json') or '{' in content[:100]:
                    try:
                        json_data = json.loads(content)
                        facts = json_data
                        logger.info(f"Successfully loaded JSON data from {txt_file}")
                        break  # Found a valid JSON, use it
                    except json.JSONDecodeError:
                        # Not valid JSON, treat as regular text
                        facts["individual_documents"][txt_file.stem] = content
                else:
                    # Regular text file
                    facts["individual_documents"][txt_file.stem] = content
                    
            except Exception as e:
                logger.error(f"Error processing text file {txt_file}: {e}")
        
        # If we have individual documents but no consolidated summary, generate one
        if facts["individual_documents"] and not facts["consolidated_summary"]:
            try:
                # Generate a consolidated summary using OpenAI
                docs_summary = "\n\n".join([f"## {doc_name}\n{content}" for doc_name, content in facts["individual_documents"].items()])
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal assistant who summarizes and organizes case facts for personal injury demand letters."},
                        {"role": "user", "content": f"Based on these extracted facts from multiple documents, create a consolidated and organized summary of the most important facts for this case. Organize by categories like incident details, injuries, treatment, damages, etc.\n\n{docs_summary}"}
                    ],
                    temperature=0.2
                )
                
                facts["consolidated_summary"] = response.choices[0].message.content
                logger.info("Generated consolidated summary from individual documents")
            except Exception as e:
                logger.error(f"Error generating consolidated summary: {e}")
                facts["consolidated_summary"] = "Unable to generate consolidated summary. Please review individual documents."
        
        # Save the extracted facts
        with open(facts_file, 'w') as f:
            json.dump(facts, f, indent=2)
        
        logger.info(f"Extracted facts saved to {facts_file}")
        return facts
    
    # Standard processing for PDF files
    logger.info("Processing source documents to extract key facts.")
    source_docs_path = Path("data/source_documents")
    source_files = list(source_docs_path.glob("*.pdf"))
    
    if not source_files:
        logger.warning("No source documents found in data/source_documents/")
        # Create a minimal facts dictionary
        facts = {
            "consolidated_summary": "No source documents were available for processing.",
            "individual_documents": {}
        }
        
        # Save the minimal facts
        with open(facts_file, 'w') as f:
            json.dump(facts, f, indent=2)
        
        return facts
    
    # Process source documents using OpenAI
    try:
        facts = process_source_documents(client, source_files)
        
        # Save extracted facts
        with open(facts_file, 'w') as f:
            json.dump(facts, f, indent=2)
        
        logger.info(f"Extracted facts saved to {facts_file}")
        return facts
    except Exception as e:
        logger.error(f"Error processing source documents: {e}")
        # Create a minimal facts dictionary with the error
        facts = {
            "consolidated_summary": f"Error processing source documents: {e}",
            "individual_documents": {}
        }
        
        # Save the minimal facts
        with open(facts_file, 'w') as f:
            json.dump(facts, f, indent=2)
        
        return facts

def evaluate_demand_letter(letter_path, facts, model="gpt-4o"):
    """
    Evaluate a single demand letter using the GPT model.
    
    Args:
        letter_path: Path to the demand letter PDF
        facts: Dictionary of extracted facts from source documents
        model: OpenAI model to use for evaluation
    
    Returns:
        Evaluation results as a dictionary
    """
    logger.info(f"Evaluating demand letter: {letter_path}")
    
    # Extract text from the demand letter
    letter_text = extract_text_from_pdf(letter_path)
    
    # Load the evaluation template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("evaluation_prompt.j2")
    
    # Prepare facts for the template
    facts_text = facts.get("consolidated_summary", "No consolidated facts available.")
    
    # Debug the facts
    logger.debug(f"Facts for evaluation: {facts_text[:500]}...")
    
    # Render the prompt
    prompt = template.render(
        source_document_facts=facts_text,
        demand_letter_content=letter_text
    )
    
    # Call OpenAI API with a stronger system message for critical evaluation
    logger.info(f"Submitting evaluation to {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert legal evaluator who specializes in assessing demand letters for personal injury cases. You have a reputation for being thorough, critical, and having very high standards. You should be strict in your evaluation and only give high scores when fully warranted by exceptional work. Apply the critical failure conditions rigorously."},
            {"role": "user", "content": prompt}
        ],
    )
    
    # Parse and return the evaluation results
    try:
        evaluation_text = response.choices[0].message.content
        
        # Extract scores and explanations with improved parsing
        # This handles various response formats from the model
        lines = evaluation_text.strip().split('\n')
        scores = {}
        
        for i, line in enumerate(lines):
            # Look for lines with category names and scores
            if ':' in line:
                category, rest = line.split(':', 1)
                
                # Clean up the category name
                category = category.strip()
                
                # Find the score (1-5)
                import re
                score_match = re.search(r'\b[1-5]\b', rest)
                
                if score_match:
                    score = int(score_match.group(0))
                    # Get explanation - everything after the score
                    explanation_text = rest[score_match.end():].strip(' -')
                    
                    # If explanation is empty, look for it in the next line
                    if not explanation_text and i+1 < len(lines):
                        explanation_text = lines[i+1].strip()
                    
                    scores[category] = {"score": score, "explanation": explanation_text}
        
        # Calculate weighted score
        weighted_score = calculate_weighted_score(scores)
        
        result = {
            "letter_name": os.path.basename(letter_path),
            "model_used": model,
            "category_scores": scores,
            "weighted_score": weighted_score,
            "full_evaluation": evaluation_text
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error parsing evaluation results: {e}")
        return {
            "letter_name": os.path.basename(letter_path),
            "error": str(e),
            "full_response": response.choices[0].message.content
        }

def compare_evaluations(evaluations):
    """
    Compare multiple demand letter evaluations.
    
    Args:
        evaluations: List of evaluation result dictionaries
    
    Returns:
        Comparison summary
    """
    if len(evaluations) < 2:
        return "Need at least two evaluations to compare."
    
    # Sort by weighted score
    sorted_evals = sorted(evaluations, key=lambda x: x.get("weighted_score", 0), reverse=True)
    
    # Generate comparison text
    comparison = "# Demand Letter Evaluation Comparison\n\n"
    comparison += "## Overall Ranking\n\n"
    
    for i, eval_result in enumerate(sorted_evals):
        comparison += f"{i+1}. {eval_result['letter_name']} - Score: {eval_result.get('weighted_score', 'N/A'):.2f}\n"
    
    comparison += "\n## Category Comparison\n\n"
    
    # Compare each category
    categories = ["Structure and Organization", "Factual Presentation", "Medical Documentation", 
                 "Damages Calculation", "Legal Strategy", "Persuasiveness", "Source Document Representation"]
    
    for category in categories:
        comparison += f"\n### {category}\n\n"
        category_sorted = sorted(evaluations, 
                               key=lambda x: x.get("category_scores", {}).get(category, {}).get("score", 0), 
                               reverse=True)
        
        for eval_result in category_sorted:
            cat_data = eval_result.get("category_scores", {}).get(category, {})
            score = cat_data.get("score", "N/A")
            explanation = cat_data.get("explanation", "No explanation provided")
            comparison += f"- {eval_result['letter_name']}: {score}/5 - {explanation}\n"
    
    return comparison

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Evaluate legal demand letters against source documents.")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing of source documents")
    parser.add_argument("--model", default="o3-2025-04-16", help="OpenAI model to use for evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare all evaluated letters")
    args = parser.parse_args()
    
    # Ensure folder structure exists
    setup_folders()
    
    # Extract facts from source documents
    facts = extract_facts_from_source_documents(force_reprocess=args.reprocess)
    
    # Get all demand letters to evaluate
    demand_letters_path = Path("data/demand_letters")
    letters = list(demand_letters_path.glob("*.pdf"))
    
    if not letters:
        logger.error("No demand letters found in data/demand_letters/")
        return
    
    evaluations = []
    
    # Evaluate each letter
    for letter_path in letters:
        result = evaluate_demand_letter(letter_path, facts, model=args.model)
        
        # Save individual evaluation result
        result_file = Path(f"data/results/{letter_path.stem}_evaluation.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Evaluation saved to {result_file}")
        evaluations.append(result)
    
    # Compare evaluations if requested
    if args.compare and len(evaluations) >= 2:
        comparison = compare_evaluations(evaluations)
        comparison_file = Path("data/results/comparison.md")
        
        with open(comparison_file, 'w') as f:
            f.write(comparison)
        
        logger.info(f"Comparison saved to {comparison_file}")
        print("\nDemand Letter Comparison Summary:")
        print("---------------------------------")
        
        # Print a simplified summary
        for eval_result in sorted(evaluations, key=lambda x: x.get("weighted_score", 0), reverse=True):
            print(f"{eval_result['letter_name']}: {eval_result.get('weighted_score', 'N/A'):.2f}")

if __name__ == "__main__":
    main()