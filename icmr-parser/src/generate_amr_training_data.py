#!/usr/bin/env python3
"""
Comprehensive AMR Training Data Generator
Generates diverse training examples for a general-purpose AMR LLM:
- General Q&A about guidelines
- Drug information queries
- Pathogen-specific queries  
- Treatment recommendations
- Prescription validation cases
"""

import json
import random
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AMRDatasetGenerator:
    def __init__(self, icmr_data_path: str):
        with open(icmr_data_path, 'r', encoding='utf-8') as f:
            self.icmr_data = json.load(f)
        logger.info(f"Loaded {len(self.icmr_data)} syndromes from ICMR data")
    
    def generate_guideline_qa(self, count: int = 50) -> List[Dict]:
        """Generate Q&A pairs about ICMR guidelines"""
        examples = []
        
        templates = [
            "What is the first-line treatment for {syndrome}?",
            "What does ICMR recommend for treating {syndrome}?",
            "Show me the treatment guidelines for {syndrome}",
            "What antibiotics are recommended for {syndrome}?",
            "List the first-line drugs for {syndrome}",
        ]
        
        for _ in range(count):
            syndrome_data = random.choice(self.icmr_data)
            syndrome = syndrome_data['syndrome_name']
            template = random.choice(templates)
            
            # Generate response
            first_line_drugs = syndrome_data.get('first_line_drugs', [])
            alternative_drugs = syndrome_data.get('alternative_drugs', [])
            
            response = f"According to ICMR 2025 Guidelines for {syndrome}:\n\n"
            
            if first_line_drugs:
                response += "**First-Line Treatment:**\n"
                for drug in first_line_drugs[:3]:  # Top 3
                    response += f"- {drug['drug_name']}"
                    if drug.get('dosage'):
                        response += f" {drug['dosage']}"
                    if drug.get('route'):
                        response += f" ({drug['route']})"
                    if drug.get('duration'):
                        response += f" for {drug['duration']}"
                    response += "\n"
            
            if alternative_drugs and len(alternative_drugs) > 0:
                response += f"\n**Alternative Options:**\n"
                for drug in alternative_drugs[:2]:
                    response += f"- {drug['drug_name']}\n"
            
            # Add pathogen info if available
            if syndrome_data.get('common_pathogens'):
                pathogen_names = [p.get('organism_name', p.get('common_name', 'Unknown')) for p in syndrome_data['common_pathogens'][:3]]
                response += f"\n**Common Pathogens:** {', '.join(pathogen_names)}"
            
            response += f"\n\n*Reference: ICMR 2025 Guidelines, Page {syndrome_data.get('page_number', 'N/A')}*"
            
            examples.append({
                "task_type": "guideline_lookup",
                "query": template.format(syndrome=syndrome),
                "response": response,
                "syndrome": syndrome,
                "source": "ICMR 2025"
            })
        
        return examples
    
    def generate_drug_info_queries(self, count: int = 30) -> List[Dict]:
        """Generate drug information queries"""
        examples = []
        
        # Collect all unique drugs
        all_drugs = set()
        for syndrome in self.icmr_data:
            first_line = syndrome.get('first_line_drugs', []) or []
            alternative = syndrome.get('alternative_drugs', []) or []
            for drug in first_line + alternative:
                all_drugs.add(drug['drug_name'])
        
        all_drugs = list(all_drugs)
        
        templates = [
            "What is {drug} used for?",
            "Tell me about {drug}",
            "What are the indications for {drug}?",
            "In which infections is {drug} used?",
            "When should I prescribe {drug}?",
        ]
        
        for _ in range(min(count, len(all_drugs))):
            drug = random.choice(all_drugs)
            template = random.choice(templates)
            
            # Find syndromes where this drug is used
            indications = []
            for syndrome_data in self.icmr_data:
                first_line = syndrome_data.get('first_line_drugs', []) or []
                alternative = syndrome_data.get('alternative_drugs', []) or []
                for drug_entry in first_line + alternative:
                    if drug_entry['drug_name'] == drug:
                        indications.append(syndrome_data['syndrome_name'])
                        break
            
            response = f"**{drug}** is used in the treatment of:\n\n"
            for indication in indications[:5]:
                response += f"- {indication}\n"
            
            response += f"\nThis antibiotic is recommended by ICMR 2025 guidelines for these infections."
            
            examples.append({
                "task_type": "drug_information",
                "query": template.format(drug=drug),
                "response": response,
                "drug": drug,
                "source": "ICMR 2025"
            })
        
        return examples
    
    def generate_pathogen_queries(self, count: int = 30) -> List[Dict]:
        """Generate pathogen-specific treatment queries"""
        examples = []
        
        # Collect all pathogens
        pathogen_to_syndromes = {}
        for syndrome_data in self.icmr_data:
            for pathogen_data in syndrome_data.get('common_pathogens', []):
                pathogen = pathogen_data.get('organism_name', pathogen_data.get('common_name', 'Unknown'))
                if pathogen and pathogen != 'Unknown':
                    if pathogen not in pathogen_to_syndromes:
                        pathogen_to_syndromes[pathogen] = []
                    pathogen_to_syndromes[pathogen].append(syndrome_data)
        
        templates = [
            "How do I treat {pathogen} infections?",
            "What antibiotics work against {pathogen}?",
            "Tell me about {pathogen} treatment",
            "What drugs are effective for {pathogen}?",
        ]
        
        for pathogen, syndromes in list(pathogen_to_syndromes.items())[:count]:
            template = random.choice(templates)
            
            # Get drugs used for this pathogen
            drugs = set()
            syndrome_names = []
            for syndrome_data in syndromes[:3]:
                syndrome_names.append(syndrome_data['syndrome_name'])
                for drug in syndrome_data.get('first_line_drugs', []):
                    drugs.add(drug['drug_name'])
            
            response = f"**{pathogen}** is commonly associated with:\n\n"
            for syndrome in syndrome_names:
                response += f"- {syndrome}\n"
            
            response += f"\n**Effective antibiotics include:**\n"
            for drug in list(drugs)[:5]:
                response += f"- {drug}\n"
            
            response += f"\n*Based on ICMR 2025 Guidelines*"
            
            examples.append({
                "task_type": "pathogen_treatment",
                "query": template.format(pathogen=pathogen),
                "response": response,
                "pathogen": pathogen,
                "source": "ICMR 2025"
            })
        
        return examples
    
    def generate_educational_content(self, count: int = 20) -> List[Dict]:
        """Generate educational Q&A about AMR concepts"""
        examples = [
            {
                "task_type": "education",
                "query": "What is antimicrobial stewardship?",
                "response": "Antimicrobial stewardship refers to coordinated interventions designed to improve and measure the appropriate use of antimicrobials by promoting the selection of the optimal antimicrobial drug regimen, dose, duration of therapy, and route of administration. Key principles include:\n\n- Using the right antibiotic for the right infection\n- Using narrow-spectrum agents when possible\n- Prescribing for appropriate durations\n- De-escalating therapy based on culture results\n- Avoiding antibiotics for viral infections\n\nThis helps prevent antimicrobial resistance and improves patient outcomes.",
                "source": "AMR education"
            },
            {
                "task_type": "education",
                "query": "What is the difference between first-line and alternative drugs?",
                "response": "**First-line drugs** are the preferred initial treatment options based on:\n- Proven efficacy\n- Safety profile\n- Cost-effectiveness\n- Local resistance patterns\n\n**Alternative drugs** are used when:\n- Patient has allergies to first-line agents\n- First-line therapy has failed\n- Culture results indicate resistance\n- Specific contraindications exist\n\nICMR guidelines provide both first-line and alternative options to guide rational antibiotic use.",
                "source": "AMR education"
            },
            {
                "task_type": "education",
                "query": "Why is antibiotic resistance a problem?",
                "response": "Antibiotic resistance is a major global health threat because:\n\n1. **Treatment failures**: Infections become harder or impossible to treat\n2. **Increased mortality**: Resistant infections lead to more deaths\n3. **Longer hospitalizations**: Patients require extended care\n4. **Higher healthcare costs**: More expensive drugs and procedures needed\n5. **Limited options**: Fewer effective antibiotics available\n6. **Spread of resistance**: Resistant genes can transfer between bacteria\n\nRational antibiotic use and adherence to guidelines like ICMR 2025 help combat resistance.",
                "source": "AMR education"
            },
            {
                "task_type": "education",
                "query": "What are ESBL-producing bacteria?",
                "response": "ESBL (Extended-Spectrum Beta-Lactamase) producing bacteria are organisms that produce enzymes capable of breaking down many common antibiotics, including:\n\n- Penicillins\n- Most cephalosporins\n- Aztreonam\n\n**Common ESBL producers:**\n- E. coli\n- Klebsiella species\n- Proteus mirabilis\n\n**Treatment options:**\n- Carbapenems (e.g., Meropenem, Imipenem)\n- Piperacillin-Tazobactam (for less severe infections)\n- Some fluoroquinolones (based on susceptibility)\n\nESBL infections require culture-guided therapy and strict infection control measures.",
                "source": "AMR education"
            },
            {
                "task_type": "education",
                "query": "What is empirical antibiotic therapy?",
                "response": "Empirical antibiotic therapy is the initial treatment given before laboratory culture and sensitivity results are available. It is based on:\n\n1. **Clinical diagnosis** of the infection\n2. **Most likely pathogens** for that infection\n3. **Local resistance patterns**\n4. **Patient factors** (age, comorbidities, allergies)\n5. **Guideline recommendations** (like ICMR 2025)\n\n**Key principle**: Start broad-spectrum therapy, then de-escalate to narrow-spectrum based on culture results (de-escalation strategy).\n\nThis approach ensures timely treatment while minimizing unnecessary broad-spectrum use.",
                "source": "AMR education"
            }
        ]
        
        return examples[:count]
    
    def generate_syndrome_overview_queries(self, count: int = 30) -> List[Dict]:
        """Generate comprehensive syndrome overview queries"""
        examples = []
        
        templates = [
            "Tell me about {syndrome}",
            "What should I know about {syndrome}?",
            "Give me an overview of {syndrome}",
            "Explain {syndrome} treatment",
        ]
        
        for _ in range(count):
            syndrome_data = random.choice(self.icmr_data)
            syndrome = syndrome_data['syndrome_name']
            template = random.choice(templates)
            
            response = f"# {syndrome}\n\n"
            
            # Definition
            if syndrome_data.get('definition'):
                response += f"**Definition:** {syndrome_data['definition']}\n\n"
            
            # Pathogens
            if syndrome_data.get('common_pathogens'):
                response += f"**Common Pathogens:**\n"
                for pathogen_data in syndrome_data['common_pathogens'][:5]:
                    pathogen = pathogen_data.get('organism_name', pathogen_data.get('common_name', 'Unknown'))
                    response += f"- {pathogen}\n"
                response += "\n"
            
            # First-line treatment
            if syndrome_data.get('first_line_drugs'):
                response += f"**First-Line Treatment (ICMR 2025):**\n"
                for drug in syndrome_data['first_line_drugs'][:3]:
                    response += f"- {drug['drug_name']}"
                    if drug.get('dosage'):
                        response += f" {drug['dosage']}"
                    if drug.get('duration'):
                        response += f" for {drug['duration']}"
                    response += "\n"
                response += "\n"
            
            # Special populations
            if syndrome_data.get('special_populations'):
                response += f"**Special Populations:** Consider dose adjustments or alternative agents\n\n"
            
            response += f"*Reference: ICMR 2025 Guidelines, Page {syndrome_data.get('page_number', 'N/A')}*"
            
            examples.append({
                "task_type": "syndrome_overview",
                "query": template.format(syndrome=syndrome),
                "response": response,
                "syndrome": syndrome,
                "source": "ICMR 2025"
            })
        
        return examples
    
    def generate_all(self, output_path: str):
        """Generate all types of training data"""
        logger.info("Generating comprehensive AMR training dataset...")
        
        all_examples = []
        
        # Generate each type
        logger.info("Generating guideline Q&A...")
        all_examples.extend(self.generate_guideline_qa(count=50))
        
        logger.info("Generating drug information queries...")
        all_examples.extend(self.generate_drug_info_queries(count=30))
        
        logger.info("Generating pathogen-specific queries...")
        all_examples.extend(self.generate_pathogen_queries(count=30))
        
        logger.info("Generating educational content...")
        all_examples.extend(self.generate_educational_content(count=20))
        
        logger.info("Generating syndrome overviews...")
        all_examples.extend(self.generate_syndrome_overview_queries(count=30))
        
        # Shuffle
        random.shuffle(all_examples)
        
        # Save as JSONL
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Generated {len(all_examples)} training examples")
        logger.info(f"✅ Saved to {output_file}")
        
        # Print statistics
        task_counts = {}
        for ex in all_examples:
            task_type = ex['task_type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        print("\n📊 Dataset Statistics:")
        print("=" * 60)
        for task_type, count in sorted(task_counts.items()):
            print(f"  {task_type:30} {count:5} examples")
        print("=" * 60)
        print(f"  TOTAL: {len(all_examples)} examples\n")
        
        return all_examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate comprehensive AMR training data')
    parser.add_argument('--icmr-data', default='../data/structured/all_syndromes_gemini.json',
                        help='Path to ICMR structured data')
    parser.add_argument('--output', default='../data/amr_training_data.jsonl',
                        help='Output path for training data')
    args = parser.parse_args()
    
    generator = AMRDatasetGenerator(args.icmr_data)
    generator.generate_all(args.output)


if __name__ == '__main__':
    main()

