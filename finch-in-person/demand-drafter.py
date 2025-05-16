#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List
from openai import OpenAI

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def encode_file_to_base64(file_path: str) -> tuple[str, str]:
    """Encode a file to base64 and determine its MIME type."""
    with open(file_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Default to binary if type can't be determined
        mime_type = "application/octet-stream"

    return base64_data, mime_type

prompt = """
Generate a professional demand letter based on the provided documents. 

The letter should be formal, assertive, and include relevant details from the documents. 

Format the response in markdown with level 2 headings (##) for each section. Ensure you write the letter in a narrative manner. 

Use the following sections IN THIS EXACT ORDER with EXACTLY THESE HEADINGS: 
- ## Statement of Facts
- ## Injuries
- ## Physical Examination
- ## Diagnoses
- ## Findings
- ## Damages
  (Include an itemized chart if possible)
- ## Future Medical Expenses
- ## Duties Under Durress and Loss of Enjoyment
- ## Per Diem Assessment
- ## Summary of Damages
- ## Similar Case Verdicts
  (Leave this section blank - it will be filled in later)
- ## ASK
  (Include a reasonable ask using the above computations to justify the ask)

If relevant to the documents provided, include the following additional sections:
- ## Imaging Summaries

IMPORTANT: Maintain consistent markdown formatting throughout and ensure the headings match exactly as specified above.
"""

def create_message_content(files: List[str]) -> List[Dict[str, Any]]:
    """Create a message content list with text and files."""
    content = [
        {
            "type": "text",
            "text": prompt,
        }
    ]

    for file_path in files:
        file_name = os.path.basename(file_path)
        base64_data, mime_type = encode_file_to_base64(file_path)

        if mime_type == "application/pdf":
            content.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": base64_data,
                    "mime_type": mime_type,
                    "filename": file_name,
                }
            )

        else:
            content.append(
                {
                    "type": "text",
                    "text": base64.b64decode(base64_data).decode("utf-8"),
                }
            )

    return content


def get_files_from_directory(directory: str) -> List[str]:
    """Get all files from a directory."""
    files = []
    for path in Path(directory).rglob("*"):
        if path.is_file():
            files.append(str(path))
    return files

def research_and_add_relevant_verdicts(letter):
    """
    Add a section that finds the most similar personal injury verdict in court
    and includes it in the final ask.
    """
    # Initialize OpenAI client with Perplexity API
    client = OpenAI(
        api_key="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",#os.environ.get("OPENAI_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    # Extract key information from the letter
    injury_types = extract_injuries(letter)
    diagnoses = extract_diagnoses(letter)
    
    # Form the search query
    search_query = (
        f"Find recent personal injury settlements or verdicts for cases involving "
        f"{injury_types} with diagnoses including {diagnoses}. "
        f"Focus on cases with large settlements and include settlement amounts, jurisdiction, and year."
    )
    
    # Create the messages for Perplexity
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal research assistant. Find relevant personal injury "
                "case verdicts that resulted in significant settlements. Provide 1-2 "
                "specific examples with case names, jurisdictions, settlement amounts, "
                "and brief case summaries."
            ),
        },
        {
            "role": "user",
            "content": search_query,
        },
    ]
    
    # Make the API call to Perplexity
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
        )
        
        # Extract the research results
        research_results = response.choices[0].message.content
        
        # Format the results into a well-structured paragraph
        similar_cases_content = format_similar_cases_paragraph(research_results)
        
        # Replace the placeholder section with our researched content
        placeholder_pattern = "## Similar Case Verdicts\n"
        if placeholder_pattern in letter:
            parts = letter.split(placeholder_pattern, 1)
            # Skip any content until the next section
            next_section_index = parts[1].find("## ")
            if next_section_index != -1:
                return parts[0] + "## Similar Case Verdicts\n\n" + similar_cases_content + "\n\n" + parts[1][next_section_index:]
            else:
                return parts[0] + "## Similar Case Verdicts\n\n" + similar_cases_content + "\n\n" + parts[1]
        else:
            # Fallback: Try to insert before ASK
            return insert_before_ask_section(letter, "## Similar Case Verdicts\n\n" + similar_cases_content + "\n\n")
        
    except Exception as e:
        print(f"Error during research: {e}")
        # Return the original letter if research fails
        return letter

def extract_injuries(letter):
    """Extract injury types from the letter."""
    # Simple implementation - look for the Injuries section
    if "## Injuries" in letter:
        injuries_section = letter.split("## Injuries")[1].split("##")[0]
        # Extract key injuries, remove common words
        injuries = ", ".join([inj.strip() for inj in injuries_section.split('\n') 
                             if inj.strip() and not inj.strip().startswith('#')])
        return injuries[:100]  # Limit length for search query
    return "personal injury"

def extract_diagnoses(letter):
    """Extract diagnoses from the letter."""
    # Simple implementation - look for the Diagnoses section
    if "## Diagnoses" in letter:
        diagnoses_section = letter.split("## Diagnoses")[1].split("##")[0]
        # Extract key diagnoses, remove common words
        diagnoses = ", ".join([diag.strip() for diag in diagnoses_section.split('\n') 
                              if diag.strip() and not diag.strip().startswith('#')])
        return diagnoses[:100]  # Limit length for search query
    return "injury"

def format_similar_cases_paragraph(research_results):
    """Format the research results into a well-structured paragraph."""
    # Add an introduction
    formatted_paragraph = "Research into similar cases reveals precedents that support our valuation "
    formatted_paragraph += "of this claim. Notably:\n\n"
    
    # Extract the most relevant parts of the research
    lines = research_results.split('\n')
    relevant_lines = []
    for line in lines:
        if any(keyword in line.lower() for keyword in 
              ['settlement', 'verdict', 'awarded', 'compensation', '$']):
            relevant_lines.append(line)
    
    # Add the relevant lines to the paragraph
    if relevant_lines:
        formatted_paragraph += '\n'.join(relevant_lines)
    else:
        # Fallback if no specific settlements were found
        formatted_paragraph += research_results
    
    # Add a concluding sentence connecting to our case
    formatted_paragraph += "\n\nThese precedents establish a framework for valuing the present claim, "
    formatted_paragraph += "which involves similar injuries and circumstances."
    
    return formatted_paragraph

def insert_before_ask_section(letter, content):
    """Insert content before the ASK section."""
    if "## ASK" in letter:
        parts = letter.split("## ASK", 1)
        return parts[0] + content + "## ASK" + parts[1]
    else:
        # If ASK section is not found, append to the end
        return letter + "\n\n" + content


def generate_demand_letter(input_dir: str, output_file: str) -> None:
    """Generate a demand letter based on files in the input directory."""
    # Load environment variables
    load_dotenv()

    # Check if OPENAI_API_KEY is set
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Get all files from directory
    files = get_files_from_directory(input_dir)
    if not files:
        print(f"No files found in {input_dir}")
        return

    print(f"Found {len(files)} files in {input_dir}")

    # Create message content
    message_content = create_message_content(files)

    # Initialize OpenAI client - passing max_tokens through model_kwargs
    model = ChatOpenAI(model="gpt-4.1")

    # Call the APIC
    print("Generating demand letter...")
    human_message = HumanMessage(content=message_content)  # type: ignore
    response = model([human_message])

    # Save the result to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return response.content
    

def main():
    """Main function to parse arguments and call the generation function."""
    parser = argparse.ArgumentParser(
        description="Generate a demand letter from files in a directory"
    )
    parser.add_argument("input_dir", help="Directory containing the input files")
    parser.add_argument("output_file", help="Path to save the output markdown file")

    args = parser.parse_args()

    response_content = generate_demand_letter(args.input_dir, args.output_file)
    response_with_verdict = research_and_add_relevant_verdicts(response_content)

    if not isinstance(response_with_verdict, str):
        response_with_verdict = str(response_with_verdict)  # type: ignore

    with open(args.output_file, "w") as f:
        f.write(response_with_verdict)

    print(f"Demand letter successfully generated and saved to {args.output_file}")


if __name__ == "__main__":
    main()