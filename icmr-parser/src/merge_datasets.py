#!/usr/bin/env python3
"""
Merge Prescription Validation and General Q&A datasets
into a unified format for LLM fine-tuning
"""

import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def merge_datasets(
    prescription_validation_file: str,
    general_qa_file: str,
    output_file: str,
    format_type: str = "unified"
):
    """Merge two datasets into unified format"""
    
    logger.info(f"Loading prescription validation data from {prescription_validation_file}")
    prescription_cases = load_jsonl(prescription_validation_file)
    logger.info(f"  ✓ Loaded {len(prescription_cases)} prescription cases")
    
    logger.info(f"Loading general Q&A data from {general_qa_file}")
    qa_examples = load_jsonl(general_qa_file)
    logger.info(f"  ✓ Loaded {len(qa_examples)} Q&A examples")
    
    unified_data = []
    
    # Convert prescription validation cases
    logger.info("Converting prescription validation cases...")
    for case in prescription_cases:
        if format_type == "unified":
            # Unified format
            unified_data.append({
                "task_type": "prescription_validation",
                "input": case['context'],
                "output": case['final_answer'],
                "reasoning": case.get('generator_reasoning_steps', []),
                "reference": case.get('icmr_reference', ''),
                "metadata": {
                    "case_id": case.get('case_id', ''),
                    "confidence_score": case.get('confidence_score', 1.0)
                }
            })
        elif format_type == "chat":
            # Chat format (for OpenAI/Gemini)
            # Create the input message
            context = case['context']
            patient_info = context.get('patient_profile', {})
            diagnosis = context.get('diagnosis', '')
            symptoms = context.get('symptoms', [])
            prescription = context.get('prescription', {})
            
            user_message = f"""Validate this prescription according to ICMR 2025 Guidelines:

**Patient:** {patient_info.get('age', 'N/A')}yo {patient_info.get('sex', 'N/A')}
**History:** {', '.join(patient_info.get('history', [])) or 'None'}
**Allergies:** {', '.join(patient_info.get('allergies', [])) or 'None'}

**Diagnosis:** {diagnosis}
**Symptoms:** {', '.join(symptoms)}

**Prescription:**
- Drug: {prescription.get('drug', 'N/A')}
- Dosage: {prescription.get('dosage', 'N/A')}
- Route: {prescription.get('route', 'N/A')}
- Frequency: {prescription.get('frequency', 'N/A')}
- Duration: {prescription.get('duration', 'N/A')}

Please provide step-by-step clinical reasoning and a final decision (Approve/Modify/Reject)."""
            
            unified_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert clinical pharmacist specializing in antimicrobial stewardship. You validate prescriptions according to ICMR 2025 Antimicrobial Treatment Guidelines, providing detailed step-by-step reasoning and evidence-based recommendations."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    },
                    {
                        "role": "assistant",
                        "content": case['final_answer']
                    }
                ],
                "metadata": {
                    "task_type": "prescription_validation",
                    "case_id": case.get('case_id', ''),
                    "reference": case.get('icmr_reference', '')
                }
            })
    
    # Convert general Q&A
    logger.info("Converting general Q&A examples...")
    for qa in qa_examples:
        if format_type == "unified":
            unified_data.append({
                "task_type": qa.get('task_type', 'general_query'),
                "input": qa['query'],
                "output": qa['response'],
                "metadata": {
                    "source": qa.get('source', 'ICMR 2025'),
                    "syndrome": qa.get('syndrome'),
                    "drug": qa.get('drug'),
                    "pathogen": qa.get('pathogen')
                }
            })
        elif format_type == "chat":
            unified_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in antimicrobial resistance (AMR) and antimicrobial treatment. You provide accurate, evidence-based information following ICMR 2025 guidelines. Your responses are educational, clear, and cite relevant guideline pages."
                    },
                    {
                        "role": "user",
                        "content": qa['query']
                    },
                    {
                        "role": "assistant",
                        "content": qa['response']
                    }
                ],
                "metadata": {
                    "task_type": qa.get('task_type', 'general_query'),
                    "source": qa.get('source', 'ICMR 2025')
                }
            })
    
    # Save merged dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in unified_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"\n✅ Merged dataset saved to {output_path}")
    logger.info(f"   Total examples: {len(unified_data)}")
    logger.info(f"   - Prescription validation: {len(prescription_cases)}")
    logger.info(f"   - General Q&A: {len(qa_examples)}")
    
    # Print task distribution
    if format_type == "unified":
        task_counts = {}
        for item in unified_data:
            task_type = item['task_type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        print("\n📊 Task Distribution:")
        print("=" * 60)
        for task_type, count in sorted(task_counts.items()):
            pct = (count / len(unified_data)) * 100
            print(f"  {task_type:30} {count:5} ({pct:5.1f}%)")
        print("=" * 60)
    
    return unified_data


def main():
    parser = argparse.ArgumentParser(description='Merge AMR training datasets')
    parser.add_argument('--prescription-data',
                        default='data/reasoning_50/reasoning_dataset_validated.jsonl',
                        help='Path to prescription validation dataset')
    parser.add_argument('--qa-data',
                        default='data/amr_training_data.jsonl',
                        help='Path to general Q&A dataset')
    parser.add_argument('--output',
                        default='data/unified_amr_dataset.jsonl',
                        help='Output path for merged dataset')
    parser.add_argument('--format',
                        choices=['unified', 'chat'],
                        default='chat',
                        help='Output format: unified (custom) or chat (OpenAI/Gemini)')
    
    args = parser.parse_args()
    
    merge_datasets(
        args.prescription_data,
        args.qa_data,
        args.output,
        args.format
    )


if __name__ == '__main__':
    main()

