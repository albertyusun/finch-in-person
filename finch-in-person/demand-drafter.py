#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------------- #
# Bootstrap – load environment and initialise API clients                     #
# --------------------------------------------------------------------------- #
load_dotenv()  # make .env values available

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY must be set (OpenAI model calls).")
if not PERPLEXITY_API_KEY:
    raise EnvironmentError("PERPLEXITY_API_KEY must be set (Perplexity search).")

openai_client      = OpenAI(api_key=OPENAI_API_KEY)
perplexity_client  = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
# --------------------------------------------------------------------------- #

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

def encode_file_to_base64(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)
    return base64_data, mime_type or "application/octet-stream"

def create_message_content(files: List[str]) -> List[Dict[str, Any]]:
    content = [{"type": "text", "text": prompt}]
    for file_path in files:
        file_name = os.path.basename(file_path)
        base64_data, mime_type = encode_file_to_base64(file_path)
        entry = (
            {
                "type": "file",
                "source_type": "base64",
                "data": base64_data,
                "mime_type": mime_type,
                "filename": file_name,
            }
            if mime_type == "application/pdf"
            else {"type": "text", "text": base64.b64decode(base64_data).decode("utf-8")}
        )
        content.append(entry)
    return content

def get_files_from_directory(directory: str) -> List[str]:
    return [str(p) for p in Path(directory).rglob("*") if p.is_file()]

def research_and_add_relevant_verdicts(letter: str) -> str:
    try:
        key_query = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract key case details for search."},
                {"role": "user", "content": letter},
            ],
        ).choices[0].message.content

        raw_case = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Find one similar case; return details only."},
                {"role": "user", "content": key_query},
            ],
        ).choices[0].message.content

        letter_prefix = letter.split("## Settlement Demand")[0]
        new_sections = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Add precedent and settlement sections."},
                {
                    "role": "user",
                    "content": f"{letter_prefix}\n\nCase info:\n{raw_case}",
                },
            ],
        ).choices[0].message.content
        return f"{letter_prefix}\n\n{new_sections}"
    except Exception:
        return letter

def generate_demand_letter(input_dir: str, output_file: str) -> str:
    files = get_files_from_directory(input_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir}")
    content = create_message_content(files)
    model = ChatOpenAI(model="gpt-4.1")
    response = model([HumanMessage(content=content)])  # type: ignore
    return response.content  # type: ignore

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a demand letter")
    parser.add_argument("input_dir", help="Directory with input files")
    parser.add_argument("output_file", help="Output markdown path")
    args = parser.parse_args()

    letter = generate_demand_letter(args.input_dir, args.output_file)
    final_letter = research_and_add_relevant_verdicts(letter)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(final_letter)
    print(f"Demand letter saved to {args.output_file}")

if __name__ == "__main__":
    main()
