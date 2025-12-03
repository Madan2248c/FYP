"""
Prepare AMR dataset for GRPO fine-tuning.

This script converts the merged AMR dataset into the format required for GRPO training.
"""

import json
from pathlib import Path
from typing import List, Dict
from datasets import Dataset


def load_amr_dataset(dataset_path: str) -> List[Dict]:
    """Load the merged AMR dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_prescription_validation_prompt(context: Dict) -> str:
    """
    Create a prompt for prescription validation task.
    
    Args:
        context: Patient case context with diagnosis, prescription, etc.
        
    Returns:
        Formatted prompt string
    """
    patient = context.get('patient_profile', {})
    diagnosis = context.get('diagnosis', 'Unknown')
    symptoms = context.get('symptoms', [])
    prescription = context.get('prescription', {})
    
    prompt = f"""**Role:** You are a clinical pharmacist with expertise in antimicrobial stewardship, trained on ICMR 2025 Antimicrobial Treatment Guidelines.

**Task:** Validate the following prescription according to ICMR 2025 guidelines. Provide step-by-step clinical reasoning and a final decision (Approved/Modify/Reject).

**Patient Information:**
- Age: {patient.get('age', 'Unknown')} years
- Sex: {patient.get('sex', 'Unknown')}
- Medical History: {', '.join(patient.get('history', [])) or 'None'}
- Allergies: {', '.join(patient.get('allergies', [])) or 'None'}

**Diagnosis:** {diagnosis}

**Presenting Symptoms:** {', '.join(symptoms) or 'Not specified'}

**Prescription to Validate:**
- Drug: {prescription.get('drug', 'Unknown')}
- Dosage: {prescription.get('dosage', 'Not specified')}
- Route: {prescription.get('route', 'Not specified')}
- Frequency: {prescription.get('frequency', 'Not specified')}
- Duration: {prescription.get('duration', 'Not specified')}

**Instructions:**
1. Provide step-by-step clinical reasoning (6-8 steps)
2. Reference ICMR 2025 guidelines
3. Give final decision: Approved/Modify/Reject with justification
4. Cite ICMR page numbers

**Output Format:**
Step 1: [Patient Assessment]
Step 2: [Diagnosis Validation]
...
Step N: [Final considerations]

Decision: [Approved/Modify/Reject]: [Detailed justification]
Reference: ICMR 2025 Guidelines, Page [X]
"""
    return prompt


def create_general_query_prompt(query: str) -> str:
    """
    Create a prompt for general AMR query task.
    
    Args:
        query: User query about AMR/antibiotics
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""**Role:** You are an expert in antimicrobial stewardship and ICMR 2025 Antimicrobial Treatment Guidelines.

**Task:** Answer the following question accurately based on ICMR 2025 guidelines.

**Question:** {query}

**Instructions:**
1. Provide a clear, accurate answer
2. Reference ICMR 2025 guidelines when applicable
3. Include relevant drug information, pathogens, or treatment protocols
4. Cite page numbers if available

**Answer:**
"""
    return prompt


def convert_to_grpo_format(amr_data: List[Dict]) -> List[Dict]:
    """
    Convert AMR dataset to GRPO training format.
    
    GRPO format:
    {
        "prompt": "...",           # Input prompt
        "reference": "...",         # Ground truth/reference answer
        "context": {...}            # Original context for evaluation
    }
    """
    grpo_data = []
    
    for item in amr_data:
        task_type = item.get('task_type', 'unknown')
        
        if task_type == 'prescription_validation':
            # Prescription validation task
            context = item.get('context', {})
            
            # Create prompt
            prompt = create_prescription_validation_prompt(context)
            
            # Get reference answer (combine reasoning + final answer)
            reasoning_steps = item.get('generator_reasoning_steps', [])
            final_answer = item.get('final_answer', '')
            icmr_ref = item.get('icmr_reference', '')
            
            reference = '\n'.join(
                [f"Step {i+1}: {step}" for i, step in enumerate(reasoning_steps)]
            )
            reference += f"\n\n{final_answer}"
            if icmr_ref:
                reference += f"\nReference: {icmr_ref}"
            
            grpo_data.append({
                'prompt': prompt,
                'reference': reference,
                'context': context,
                'task_type': 'prescription_validation',
                'case_id': item.get('case_id', 'unknown')
            })
            
        elif task_type in ['guideline_lookup', 'drug_information', 'pathogen_treatment', 
                           'educational_content', 'syndrome_overview', 'general_query']:
            # General query task
            query = item.get('query', '') or item.get('input', '')
            response = item.get('response', '') or item.get('output', '')
            
            if query and response:
                prompt = create_general_query_prompt(query)
                
                grpo_data.append({
                    'prompt': prompt,
                    'reference': response,
                    'context': {'query': query},
                    'task_type': task_type,
                    'case_id': item.get('case_id', 'unknown')
                })
    
    return grpo_data


def save_grpo_dataset(data: List[Dict], output_path: str):
    """Save GRPO dataset to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(data)} examples to {output_path}")


def create_huggingface_dataset(data: List[Dict]) -> Dataset:
    """Convert to HuggingFace Dataset format."""
    return Dataset.from_list(data)


def main():
    """Main execution function."""
    print("=" * 80)
    print("AMR Dataset Preparation for GRPO Training")
    print("=" * 80)
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "amr_merged_final_dataset.jsonl"
    output_dir = base_dir / "grpo_training" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"📂 Loading dataset from: {input_file}")
    amr_data = load_amr_dataset(str(input_file))
    print(f"✅ Loaded {len(amr_data)} examples")
    
    # Convert to GRPO format
    print("\n🔄 Converting to GRPO format...")
    grpo_data = convert_to_grpo_format(amr_data)
    print(f"✅ Converted {len(grpo_data)} examples")
    
    # Statistics
    task_counts = {}
    for item in grpo_data:
        task_type = item['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print("\n📊 Dataset Statistics:")
    print("-" * 80)
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count} examples")
    print("-" * 80)
    print(f"  Total: {len(grpo_data)} examples")
    
    # Split into train/val
    train_size = int(0.95 * len(grpo_data))
    train_data = grpo_data[:train_size]
    val_data = grpo_data[train_size:]
    
    print(f"\n📊 Split: {len(train_data)} train, {len(val_data)} validation")
    
    # Save datasets
    print("\n💾 Saving datasets...")
    save_grpo_dataset(train_data, str(output_dir / "train_grpo.json"))
    save_grpo_dataset(val_data, str(output_dir / "val_grpo.json"))
    save_grpo_dataset(grpo_data, str(output_dir / "full_grpo.json"))
    
    # Create HuggingFace datasets
    train_dataset = create_huggingface_dataset(train_data)
    val_dataset = create_huggingface_dataset(val_data)
    
    # Save to HuggingFace format
    train_dataset.save_to_disk(str(output_dir / "train_hf"))
    val_dataset.save_to_disk(str(output_dir / "val_hf"))
    
    print(f"✅ Saved HuggingFace datasets to {output_dir}")
    
    # Show example
    print("\n" + "=" * 80)
    print("Example Training Instance:")
    print("=" * 80)
    example = train_data[0]
    print(f"\n📝 Prompt (first 500 chars):")
    print(example['prompt'][:500] + "...")
    print(f"\n✨ Reference (first 300 chars):")
    print(example['reference'][:300] + "...")
    print(f"\n🏷️  Task Type: {example['task_type']}")
    print("=" * 80)
    
    print("\n✅ Dataset preparation complete!")
    print(f"📁 Output directory: {output_dir}")
    print("\nNext steps:")
    print("1. Deploy Supabase Edge Function for evaluation")
    print("2. Run GRPO training with grpo_train_amr.py")


if __name__ == "__main__":
    main()

