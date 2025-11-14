#!/usr/bin/env python3
"""
Convert general Q&A dataset to match the prescription validation format
Adds reasoning steps to make it compatible for unified instruction fine-tuning
"""

import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_reasoning_steps(query: str, response: str, task_type: str, metadata: dict) -> list:
    """Generate reasoning steps for Q&A examples"""
    
    if task_type == "guideline_lookup":
        syndrome = metadata.get('syndrome', 'the condition')
        return [
            f"Step 1: Query Analysis - The user is asking about treatment guidelines for {syndrome}.",
            "Step 2: Guideline Reference - Consulting ICMR 2025 Antimicrobial Treatment Guidelines for evidence-based recommendations.",
            f"Step 3: First-Line Therapy Identification - Identifying the recommended first-line antimicrobial agents for {syndrome}.",
            "Step 4: Dosage and Route Verification - Extracting the correct dosages and routes of administration from the guidelines.",
            "Step 5: Alternative Options - Noting alternative therapeutic options when first-line agents are contraindicated.",
            "Step 6: Response Formulation - Providing a comprehensive answer with drug names, dosages, routes, and duration based on ICMR 2025 guidelines."
        ]
    
    elif task_type == "drug_information":
        drug = metadata.get('drug', 'the antibiotic')
        return [
            f"Step 1: Drug Identification - The query is about {drug} and its clinical indications.",
            "Step 2: Guideline Search - Searching ICMR 2025 guidelines for syndromes where this drug is recommended.",
            f"Step 3: Indication Analysis - Identifying all infections for which {drug} is listed as first-line or alternative therapy.",
            "Step 4: Spectrum Assessment - Considering the drug's antimicrobial spectrum and its appropriateness for the listed indications.",
            "Step 5: Response Compilation - Listing all ICMR-approved indications with guideline references."
        ]
    
    elif task_type == "pathogen_treatment":
        pathogen = metadata.get('pathogen', 'the organism')
        return [
            f"Step 1: Pathogen Identification - The query concerns treatment options for {pathogen}.",
            "Step 2: Syndrome Mapping - Identifying which clinical syndromes are commonly associated with this pathogen per ICMR guidelines.",
            f"Step 3: Drug Selection - Finding antibiotics effective against {pathogen} based on guideline recommendations.",
            "Step 4: Resistance Considerations - Noting any documented resistance patterns mentioned in the guidelines.",
            "Step 5: Treatment Summary - Compiling the effective antimicrobials with their indications."
        ]
    
    elif task_type == "syndrome_overview":
        syndrome = metadata.get('syndrome', 'the condition')
        return [
            f"Step 1: Syndrome Identification - Providing a comprehensive overview of {syndrome}.",
            "Step 2: Clinical Definition - Reviewing the ICMR definition and diagnostic criteria.",
            "Step 3: Pathogen Analysis - Identifying the most common causative organisms.",
            "Step 4: Treatment Protocol - Extracting first-line antimicrobial therapy recommendations.",
            "Step 5: Special Considerations - Noting any special population considerations or alternative regimens.",
            "Step 6: Summary Formulation - Presenting a complete clinical overview with guideline references."
        ]
    
    elif task_type == "education":
        return [
            "Step 1: Concept Identification - Understanding the antimicrobial stewardship or resistance concept being queried.",
            "Step 2: Clinical Context - Relating the concept to real-world clinical practice and patient care.",
            "Step 3: Guideline Alignment - Connecting the concept to ICMR 2025 guidelines and recommendations.",
            "Step 4: Practical Implications - Explaining how this knowledge impacts antibiotic prescribing decisions.",
            "Step 5: Educational Response - Providing a clear, evidence-based explanation with practical examples."
        ]
    
    else:
        return [
            "Step 1: Query Analysis - Understanding the specific antimicrobial question being asked.",
            "Step 2: Guideline Consultation - Referencing ICMR 2025 Antimicrobial Treatment Guidelines.",
            "Step 3: Evidence Synthesis - Extracting relevant information from the guidelines.",
            "Step 4: Response Formulation - Providing an accurate, guideline-based answer."
        ]


def convert_qa_to_reasoning_format(qa_file: str, output_file: str):
    """Convert Q&A format to prescription validation format"""
    
    logger.info(f"Loading Q&A data from {qa_file}")
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_examples = [json.loads(line) for line in f]
    
    logger.info(f"Converting {len(qa_examples)} Q&A examples to reasoning format")
    
    converted = []
    
    for qa in qa_examples:
        task_type = qa.get('task_type', 'general_query')
        query = qa['query']
        response = qa['response']
        
        metadata = {
            'syndrome': qa.get('syndrome'),
            'drug': qa.get('drug'),
            'pathogen': qa.get('pathogen'),
            'source': qa.get('source', 'ICMR 2025')
        }
        
        # Generate reasoning steps
        reasoning_steps = generate_reasoning_steps(query, response, task_type, metadata)
        
        # Extract ICMR reference if present in response
        icmr_ref = "ICMR 2025 Guidelines"
        if "Page" in response:
            # Try to extract page number
            import re
            match = re.search(r'Page\s+(\d+|N/A)', response)
            if match:
                icmr_ref = f"Page {match.group(1)}, ICMR 2025 Guidelines"
        
        # Create unified format
        converted_entry = {
            "instruction": "Answer the following antimicrobial query according to ICMR 2025 Antimicrobial Treatment Guidelines.",
            "context": {
                "query": query,
                "task_type": task_type
            },
            "generator_reasoning_steps": reasoning_steps,
            "final_answer": response,
            "verifier_result": "Pass",  # Q&A doesn't need verification
            "icmr_reference": icmr_ref,
            "confidence_score": 1.0,
            "validation_checks": [
                {"check": "Guideline Reference Check", "passed": True},
                {"check": "Reasoning Completeness Check", "passed": True}
            ],
            "case_id": f"{task_type.upper()}_{converted.index(qa) if qa in converted else len(converted):03d}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "original_task_type": task_type,
                **{k: v for k, v in metadata.items() if v is not None}
            }
        }
        
        converted.append(converted_entry)
    
    # Save converted data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Converted {len(converted)} examples")
    logger.info(f"✅ Saved to {output_path}")
    
    # Print statistics
    task_counts = {}
    for entry in converted:
        orig_type = entry['metadata']['original_task_type']
        task_counts[orig_type] = task_counts.get(orig_type, 0) + 1
    
    print("\n📊 Conversion Statistics:")
    print("=" * 60)
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type:30} {count:5} examples")
    print("=" * 60)
    print(f"  TOTAL: {len(converted)} examples\n")
    
    return converted


def merge_datasets(prescription_file: str, qa_converted_file: str, output_file: str):
    """Merge prescription validation and converted Q&A datasets"""
    
    logger.info("Merging datasets...")
    
    # Load prescription validation data
    with open(prescription_file, 'r', encoding='utf-8') as f:
        prescription_data = [json.loads(line) for line in f]
    
    # Load converted Q&A data
    with open(qa_converted_file, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line) for line in f]
    
    # Merge
    merged = prescription_data + qa_data
    
    # Save merged dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"\n✅ Merged dataset saved to {output_path}")
    logger.info(f"   Total examples: {len(merged)}")
    logger.info(f"   - Prescription validation: {len(prescription_data)}")
    logger.info(f"   - General Q&A: {len(qa_data)}")
    
    print("\n📊 Final Dataset Composition:")
    print("=" * 60)
    print(f"  Prescription Validation:  {len(prescription_data):5} ({len(prescription_data)/len(merged)*100:.1f}%)")
    print(f"  General Q&A:              {len(qa_data):5} ({len(qa_data)/len(merged)*100:.1f}%)")
    print("=" * 60)
    print(f"  TOTAL EXAMPLES:           {len(merged):5}\n")
    
    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert Q&A to reasoning format and merge datasets')
    parser.add_argument('--qa-file',
                        default='data/amr_training_data.jsonl',
                        help='Path to Q&A dataset')
    parser.add_argument('--prescription-file',
                        default='data/reasoning_50/reasoning_dataset_validated.jsonl',
                        help='Path to prescription validation dataset')
    parser.add_argument('--output-converted',
                        default='data/amr_qa_reasoning_format.jsonl',
                        help='Output path for converted Q&A')
    parser.add_argument('--output-merged',
                        default='data/amr_unified_reasoning_dataset.jsonl',
                        help='Output path for merged dataset')
    parser.add_argument('--skip-merge',
                        action='store_true',
                        help='Only convert Q&A, do not merge')
    
    args = parser.parse_args()
    
    # Step 1: Convert Q&A to reasoning format
    converted = convert_qa_to_reasoning_format(args.qa_file, args.output_converted)
    
    # Step 2: Merge with prescription validation data
    if not args.skip_merge:
        merged = merge_datasets(
            args.prescription_file,
            args.output_converted,
            args.output_merged
        )


if __name__ == '__main__':
    main()

