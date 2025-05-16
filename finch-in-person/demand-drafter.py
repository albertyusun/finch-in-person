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

# Initialize OpenAI client - Move this to the top of the file
api_key = os.environ.get("PERPLEXITY_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable must be set")
client = OpenAI(api_key=api_key)  # Now properly initialized with error handling


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
Generate a professional, narrative-driven demand letter based on the provided documents. 

The letter should read like a compelling story that illustrates the client's experience while maintaining legal professionalism. Use vivid language that helps the reader understand the full human impact of the incident, not just clinical facts. Write in a flowing narrative style rather than a bullet-point or strictly clinical format.

Format the response in markdown with level 2 headings (##) for each section. The letter should tell a cohesive story across all sections, with natural transitions between topics.

Use the following sections IN THIS EXACT ORDER with EXACTLY THESE HEADINGS:

- ## Statement of Facts
  (Create a vivid narrative of the incident, including all relevant details about the parties involved, the circumstances, and clear establishment of liability)

- ## Injuries and Treatment Journey
  (Describe the injuries in narrative form, emphasizing the human experience and progressive nature of the client's medical journey)

- ## Medical Findings and Diagnoses
  (Present medical evidence and diagnoses in a coherent narrative that explains their significance to the client's experience)

- ## Impact on Daily Life
  (Create a detailed picture of how the injuries have affected the client's work, family life, and emotional well-being)

- ## Economic Damages
  (Present all economic damages in narrative form with an itemized chart embedded within this narrative)

- ## Ongoing and Future Care Needs
  (Explain future treatment requirements and their anticipated impact)

- ## Comparable Case Precedent
  (Include ONLY ONE strong, relevant case precedent with similar injuries and a favorable outcome. Provide specific details about the case, the injuries involved, and the settlement/verdict amount. Explain how this precedent directly relates to the current case.)

- ## Settlement Demand
  (Present a clear, justified settlement demand that references the case precedent as supporting evidence. The demand should be approximately 30-40% higher than the total calculated damages to allow room for negotiation.)

If relevant to the documents provided, subtly integrate the following information within the appropriate sections rather than creating separate headings:
- Physical examination details
- Imaging results
- Per diem calculations for pain and suffering

IMPORTANT:
- Maintain natural language flow throughout
- Use descriptive, impactful language without exaggeration
- Create clear connections between medical facts and human impact
- Ensure the chosen case precedent directly supports the settlement demand
- Format all headings consistently as level 2 (##) markdown headings
- Make the settlement demand compelling by tying together all previous sections
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
    Add a section that finds a similar verdict in court and includes it in the final ask.
    """
    try:
        # Initialize OpenAI client with Perplexity API for research
        perplexity_client = OpenAI(
            api_key=os.environ.get("PERPLEXITY_API_KEY", "your-perplexity-key-here"),
            base_url="https://api.perplexity.ai"
        )
        
        # Initialize regular OpenAI client for the final integration
        openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Use OpenAI to extract key details from the letter
        extract_messages = [
            {
                "role": "system",
                "content": "Extract key case details from this letter to create a search query."
            },
            {
                "role": "user",
                "content": f"Read this letter and extract only the key injuries, diagnoses, and circumstances to create a brief search query to find one similar case precedent:\n\n{letter}"
            }
        ]
        
        extract_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster model is sufficient for this task
            messages=extract_messages
        )
        
        key_details = extract_response.choices[0].message.content
        
        # Get case information from Perplexity with a generic query
        perplexity_messages = [
            {
                "role": "system",
                "content": "You are a legal researcher. Find ONE relevant case similar to the one described. Format as a direct paragraph without any introductions."
            },
            {
                "role": "user",
                "content": f"Find ONE legal case settlement similar to this one: {key_details}. Include case name, jurisdiction, settlement amount, year, and brief case summary. Just provide the information directly without any introduction."
            }
        ]
        
        perplexity_response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=perplexity_messages
        )
        
        raw_case_info = perplexity_response.choices[0].message.content
        
        # Find where to split the letter
        split_markers = ["## Similar Case Verdicts", "## Comparable Case Precedent", "## ASK", "## Settlement Demand"]
        letter_prefix = letter
        for marker in split_markers:
            if marker in letter:
                letter_prefix = letter.split(marker)[0]
                break
        
        # Have OpenAI complete the letter
        completion_messages = [
            {
                "role": "system",
                "content": "You are an expert legal writer. Complete this demand letter with two final sections."
            },
            {
                "role": "user",
                "content": f"Complete this demand letter:\n\n{letter_prefix}\n\nCase precedent info:\n{raw_case_info}\n\nWrite two sections: '## Comparable Case Precedent' describing just this ONE case and its relevance, and '## Settlement Demand' that references the case to justify a settlement 35-40% higher than the damages in the letter. Match the letter's existing narrative style."
            }
        ]
        
        completion_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=completion_messages
        )
        
        new_sections = completion_response.choices[0].message.content
        
        # Return the combined letter
        return letter_prefix + "\n\n" + new_sections
        
    except Exception as e:
        print(f"Error during research or integration: {e}")
        return letter  # Return original letter if anything fails

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