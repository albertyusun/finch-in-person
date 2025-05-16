#!/usr/bin/env python3
import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

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


def create_message_content(files: List[str]) -> List[Dict[str, Any]]:
    """Create a message content list with text and files."""
    content = [
        {
            "type": "text",
            "text": "Generate a professional demand letter based on the provided documents. The letter should be formal, assertive, and include relevant details from the documents. Format the response in markdown.",
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

    # Call the API
    print("Generating demand letter...")
    human_message = HumanMessage(content=message_content)  # type: ignore
    response = model([human_message])

    # Save the result to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the content as a string
    response_content = response.content
    if not isinstance(response_content, str):
        response_content = str(response_content)  # type: ignore

    with open(output_file, "w") as f:
        f.write(response_content)

    print(f"Demand letter successfully generated and saved to {output_file}")


def main():
    """Main function to parse arguments and call the generation function."""
    parser = argparse.ArgumentParser(
        description="Generate a demand letter from files in a directory"
    )
    parser.add_argument("input_dir", help="Directory containing the input files")
    parser.add_argument("output_file", help="Path to save the output markdown file")

    args = parser.parse_args()

    generate_demand_letter(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()