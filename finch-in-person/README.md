# Demand Letter Evaluator

A tool for evaluating legal demand letters for personal injury cases against source documents using OpenAI models.

## Overview

This tool analyzes demand letters by comparing them against source documents like medical records, police reports, and other case materials. It evaluates letters based on a weighted rubric and provides scores and explanations for each category.

## Setup

1. Clone this repository
2. Install requirements:
   ```
   pip install openai jinja2
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

## Directory Structure

```
demand-letter-evaluator/
│
├── main.py                   # Main script for running evaluations
├── utils.py                  # Utility functions (PDF processing, scoring calculations)
│
├── templates/
│   └── evaluation_prompt.j2  # Jinja template for evaluation prompts
│
├── data/
│   ├── source_documents/     # Original PDFs of source materials
│   ├── demand_letters/       # Demand letters to evaluate
│   ├── extracted_facts/      # Processed facts from source documents
│   └── results/              # Evaluation results
│
└── README.md                 # This file
```

## Usage

1. Place your source documents (PDFs) in `data/source_documents/`
2. Place demand letters to evaluate (PDFs) in `data/demand_letters/`
3. Run the evaluator:
   ```
   python main.py
   ```

### Optional Arguments

- `--reprocess`: Force reprocessing of source documents
- `--model`: Specify which OpenAI model to use (default: gpt-4o)
- `--compare`: Generate a comparison report for all evaluated letters

Example:
```
python main.py --reprocess --model gpt-4o --compare
```

## Evaluation Criteria

Letters are evaluated on a scale of 1-5 across seven categories:

1. **Structure and Organization** (15%): Organization, flow, and formatting
2. **Factual Presentation** (15%): Accuracy and clarity of incident details
3. **Medical Documentation** (20%): Thoroughness of injury and treatment descriptions
4. **Damages Calculation** (20%): Completeness and justification of damage claims
5. **Legal Strategy** (10%): Effectiveness of liability arguments and negotiation approach
6. **Persuasiveness** (10%): Professional tone and compelling presentation
7. **Source Document Representation** (10%): Accuracy compared to source documents

## Output

The tool generates:
- Individual evaluation reports for each letter (JSON)
- Comparison report when multiple letters are evaluated (Markdown)

## Notes

- PDF text extraction is simplified in this implementation. For production use, implement a proper PDF extraction method.
- The OpenAI model may incur costs based on the number and size of documents processed.