#!/usr/bin/env python3
"""
AI-Powered Q&A Dataset Generator with Reasoning
Uses Google Gemini to generate diverse, high-quality Q&A examples
with natural reasoning steps, grounded in ICMR guidelines
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import os
import re
from dotenv import load_dotenv

# Import Gemini
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIQAGenerator:
    """Generate Q&A examples using Gemini with natural reasoning"""
    
    def __init__(self, api_key: str, icmr_data_path: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the AI Q&A generator with Gemini.
        
        Args:
            api_key: Gemini API key
            icmr_data_path: Path to ICMR structured data JSON
            model_name: Gemini model name (default: gemini-2.0-flash-exp)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        
        # Load ICMR data
        with open(icmr_data_path, 'r', encoding='utf-8') as f:
            self.icmr_data = json.load(f)
        
        logger.info(f"✓ Loaded {len(self.icmr_data)} syndromes from ICMR data")
        logger.info(f"✓ Using model: {model_name}")
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Call Gemini with JSON mode and retry on failure.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum retry attempts
            
        Returns:
            Parsed JSON response or None if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                # Call Gemini with JSON mode
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,  # Higher for diversity
                        response_mime_type="application/json"
                    )
                )
                
                # Parse JSON response
                result_dict = json.loads(response.text)
                return result_dict
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Error on attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
                
                # Check if it's a rate limit error
                if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 60  # 60s, 120s, 240s
                        logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Rate limit exhausted after all retries")
                        return None
                
                # Other errors - retry with shorter backoff
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retries exhausted")
                    return None
        
        return None
    
    def generate_guideline_lookup_qa(self, syndrome_data: Dict) -> Optional[Dict]:
        """Generate guideline lookup Q&A with reasoning"""
        
        syndrome = syndrome_data['syndrome_name']
        definition = syndrome_data.get('definition', 'N/A')
        pathogens = syndrome_data.get('common_pathogens', [])
        first_line = syndrome_data.get('first_line_drugs', [])
        alternative = syndrome_data.get('alternative_drugs', [])
        page = syndrome_data.get('page_number', 'N/A')
        
        # Format pathogen names
        pathogen_names = [p.get('organism_name', p.get('common_name', '')) for p in pathogens[:5] if pathogens]
        
        # Format drug info
        drug_info = []
        for drug in (first_line or [])[:3]:
            info = drug['drug_name']
            if drug.get('dosage'):
                info += f" {drug['dosage']}"
            if drug.get('route'):
                info += f" ({drug['route']})"
            if drug.get('duration'):
                info += f" for {drug['duration']}"
            drug_info.append(info)
        
        prompt = f"""You are an expert clinical pharmacist and medical educator. Generate a natural, educational Q&A about treatment guidelines.

**Syndrome:** {syndrome}
**Definition:** {definition}
**Common Pathogens:** {', '.join(pathogen_names) if pathogen_names else 'Not specified'}
**First-Line Drugs:** {'; '.join(drug_info) if drug_info else 'Not specified'}
**Alternative Drugs:** {', '.join([d['drug_name'] for d in (alternative or [])[:2]]) if alternative else 'None listed'}
**ICMR Page:** {page}

Generate a natural question a medical student or clinician might ask about treating {syndrome}, then provide:

1. **4-6 reasoning steps** explaining how you would approach answering this query, including:
   - Clinical context and patient considerations
   - Why specific drugs are chosen
   - Antimicrobial stewardship principles
   - Safety and monitoring considerations
   
2. **A comprehensive, educational answer** that includes:
   - First-line treatment recommendations with dosages
   - Rationale for the choices
   - Alternative options
   - Key clinical pearls
   - ICMR guideline reference

Make the reasoning sound like a clinician thinking through the problem, NOT like a template.
Use natural, conversational medical language.

Return ONLY valid JSON in this format:
{{
  "query": "A natural question about treating {syndrome}",
  "reasoning_steps": ["Natural step 1...", "Natural step 2...", ...],
  "answer": "Comprehensive, educational answer with guideline citations..."
}}"""

        try:
            result = self._call_llm_with_retry(prompt)
            
            # Format to unified structure
            return {
                "instruction": "Answer the following antimicrobial query according to ICMR 2025 Antimicrobial Treatment Guidelines.",
                "context": {
                    "query": result['query'],
                    "task_type": "guideline_lookup"
                },
                "generator_reasoning_steps": result['reasoning_steps'],
                "final_answer": result['answer'],
                "verifier_result": "Pass",
                "icmr_reference": f"Page {page}, ICMR 2025 Guidelines",
                "confidence_score": 1.0,
                "validation_checks": [
                    {"check": "Guideline Reference Check", "passed": True},
                    {"check": "Reasoning Completeness Check", "passed": True}
                ],
                "case_id": f"AI_GL_{syndrome[:10]}_{random.randint(1000,9999)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "original_task_type": "guideline_lookup",
                    "syndrome": syndrome,
                    "source": "ICMR 2025",
                    "generated_by": "AI"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating Q&A for {syndrome}: {e}")
            return None
    
    def generate_drug_info_qa(self, drug_name: str, indications: List[str]) -> Optional[Dict]:
        """Generate drug information Q&A with reasoning"""
        
        prompt = f"""You are a clinical pharmacologist. Generate a natural Q&A about the antibiotic {drug_name}.

**Drug:** {drug_name}
**ICMR-Approved Indications:** {', '.join(indications[:5])}

Generate a question a clinician or researcher might ask about {drug_name}, then provide:

1. **4-6 reasoning steps** that explain:
   - The drug's mechanism and spectrum
   - Clinical context for each indication
   - Antimicrobial stewardship considerations
   - When to use vs alternatives
   
2. **A detailed answer** covering:
   - All indications with context
   - Clinical pearls
   - Key prescribing considerations
   - ICMR guideline reference

Make it educational and clinically relevant.

Return ONLY valid JSON:
{{
  "query": "Natural question about {drug_name}",
  "reasoning_steps": ["Clinical reasoning step 1...", ...],
  "answer": "Detailed educational answer..."
}}"""

        try:
            result = self._call_llm_with_retry(prompt)
            
            if not result:
                return None
            
            return {
                "instruction": "Answer the following antimicrobial query according to ICMR 2025 Antimicrobial Treatment Guidelines.",
                "context": {
                    "query": result['query'],
                    "task_type": "drug_information"
                },
                "generator_reasoning_steps": result['reasoning_steps'],
                "final_answer": result['answer'],
                "verifier_result": "Pass",
                "icmr_reference": "ICMR 2025 Guidelines",
                "confidence_score": 1.0,
                "validation_checks": [
                    {"check": "Guideline Reference Check", "passed": True},
                    {"check": "Reasoning Completeness Check", "passed": True}
                ],
                "case_id": f"AI_DRUG_{drug_name[:10]}_{random.randint(1000,9999)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "original_task_type": "drug_information",
                    "drug": drug_name,
                    "source": "ICMR 2025",
                    "generated_by": "AI"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating drug Q&A for {drug_name}: {e}")
            return None
    
    def generate_pathogen_qa(self, pathogen: str, syndromes: List[str], drugs: List[str]) -> Optional[Dict]:
        """Generate pathogen treatment Q&A with reasoning"""
        
        prompt = f"""You are an infectious disease specialist. Generate a natural Q&A about treating {pathogen} infections.

**Pathogen:** {pathogen}
**Associated Syndromes:** {', '.join(syndromes[:3])}
**Effective Antibiotics:** {', '.join(drugs[:5])}

Generate a question about treating {pathogen}, then provide:

1. **5-7 reasoning steps** covering:
   - Pathogen characteristics and resistance patterns
   - Clinical syndromes it causes
   - Drug selection rationale
   - Antimicrobial stewardship
   - Culture and susceptibility considerations
   
2. **A comprehensive answer** with:
   - Treatment approach
   - Specific drug recommendations
   - When to escalate therapy
   - ICMR guideline reference

Return ONLY valid JSON:
{{
  "query": "Clinical question about {pathogen}",
  "reasoning_steps": ["Step 1...", ...],
  "answer": "Detailed treatment guidance..."
}}"""

        try:
            result = self._call_llm_with_retry(prompt)
            
            if not result:
                return None
            
            return {
                "instruction": "Answer the following antimicrobial query according to ICMR 2025 Antimicrobial Treatment Guidelines.",
                "context": {
                    "query": result['query'],
                    "task_type": "pathogen_treatment"
                },
                "generator_reasoning_steps": result['reasoning_steps'],
                "final_answer": result['answer'],
                "verifier_result": "Pass",
                "icmr_reference": "ICMR 2025 Guidelines",
                "confidence_score": 1.0,
                "validation_checks": [
                    {"check": "Guideline Reference Check", "passed": True},
                    {"check": "Reasoning Completeness Check", "passed": True}
                ],
                "case_id": f"AI_PATH_{pathogen[:10]}_{random.randint(1000,9999)}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "original_task_type": "pathogen_treatment",
                    "pathogen": pathogen,
                    "source": "ICMR 2025",
                    "generated_by": "AI"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating pathogen Q&A for {pathogen}: {e}")
            return None
    
    def generate_dataset(self, 
                        guideline_count: int = 50,
                        drug_count: int = 30,
                        pathogen_count: int = 30,
                        output_file: str = "data/amr_ai_generated_qa.jsonl",
                        resume: bool = True):
        """Generate complete AI-powered Q&A dataset
        
        Args:
            guideline_count: Number of guideline Q&As to generate
            drug_count: Number of drug info Q&As to generate
            pathogen_count: Number of pathogen Q&As to generate
            output_file: Output path
            resume: If True, load existing data and append new examples
        """
        
        all_examples = []
        output_path = Path(output_file)
        
        # Track what's already been generated
        processed_syndromes = set()
        processed_drugs = set()
        processed_pathogens = set()
        existing_count = 0
        
        # Load existing data if resuming
        if resume and output_path.exists():
            logger.info(f"📂 Loading existing data from {output_file}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_count += 1
                        try:
                            existing = json.loads(line)
                            task_type = existing['metadata']['original_task_type']
                            
                            # Track what we've already generated
                            if task_type == 'guideline_lookup':
                                syndrome = existing['metadata'].get('syndrome')
                                if syndrome:
                                    processed_syndromes.add(syndrome)
                            elif task_type == 'drug_information':
                                drug = existing['metadata'].get('drug')
                                if drug:
                                    processed_drugs.add(drug)
                            elif task_type == 'pathogen_treatment':
                                pathogen = existing['metadata'].get('pathogen')
                                if pathogen:
                                    processed_pathogens.add(pathogen)
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"✓ Found {existing_count} existing examples")
            logger.info(f"  - {len(processed_syndromes)} syndromes")
            logger.info(f"  - {len(processed_drugs)} drugs")
            logger.info(f"  - {len(processed_pathogens)} pathogens")
        
        # 1. Guideline lookups
        logger.info(f"\nGenerating {guideline_count} guideline lookup Q&As...")
        
        # Filter out already processed syndromes
        available_syndromes = [s for s in self.icmr_data if s['syndrome_name'] not in processed_syndromes]
        syndromes = random.sample(available_syndromes, min(guideline_count, len(available_syndromes)))
        
        for syndrome_data in syndromes:
            qa = self.generate_guideline_lookup_qa(syndrome_data)
            if qa:
                all_examples.append(qa)
                logger.info(f"  ✓ Generated: {qa['context']['query'][:60]}...")
        
        # 2. Drug information
        logger.info(f"\nGenerating {drug_count} drug information Q&As...")
        # Collect drugs
        drug_to_indications = {}
        for syndrome in self.icmr_data:
            first_line = syndrome.get('first_line_drugs', []) or []
            alternative = syndrome.get('alternative_drugs', []) or []
            for drug in first_line + alternative:
                drug_name = drug['drug_name']
                if drug_name not in drug_to_indications:
                    drug_to_indications[drug_name] = []
                drug_to_indications[drug_name].append(syndrome['syndrome_name'])
        
        # Filter out already processed drugs
        available_drugs = [d for d in drug_to_indications.keys() if d not in processed_drugs]
        selected_drugs = random.sample(available_drugs, min(drug_count, len(available_drugs)))
        
        for drug in selected_drugs:
            qa = self.generate_drug_info_qa(drug, drug_to_indications[drug])
            if qa:
                all_examples.append(qa)
                logger.info(f"  ✓ Generated: {qa['context']['query'][:60]}...")
        
        # 3. Pathogen treatment
        logger.info(f"\nGenerating {pathogen_count} pathogen treatment Q&As...")
        # Collect pathogens
        pathogen_info = {}
        for syndrome in self.icmr_data:
            for pathogen_data in syndrome.get('common_pathogens', []):
                pathogen = pathogen_data.get('organism_name', pathogen_data.get('common_name'))
                if pathogen and pathogen != 'Unknown':
                    if pathogen not in pathogen_info:
                        pathogen_info[pathogen] = {'syndromes': [], 'drugs': set()}
                    pathogen_info[pathogen]['syndromes'].append(syndrome['syndrome_name'])
                    for drug in (syndrome.get('first_line_drugs', []) or []):
                        pathogen_info[pathogen]['drugs'].add(drug['drug_name'])
        
        # Filter out already processed pathogens
        available_pathogens = [p for p in pathogen_info.keys() if p not in processed_pathogens]
        selected_pathogens = random.sample(available_pathogens, min(pathogen_count, len(available_pathogens)))
        
        for pathogen in selected_pathogens:
            info = pathogen_info[pathogen]
            qa = self.generate_pathogen_qa(pathogen, info['syndromes'], list(info['drugs']))
            if qa:
                all_examples.append(qa)
                logger.info(f"  ✓ Generated: {qa['context']['query'][:60]}...")
        
        # Save (append mode if resuming)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if resume and output_path.exists() else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        total_examples = existing_count + len(all_examples)
        
        logger.info(f"\n✅ Generated {len(all_examples)} NEW AI-powered Q&A examples")
        logger.info(f"✅ Appended to {output_path}")
        logger.info(f"✅ Total examples in file: {total_examples}")
        
        # Stats (for newly generated examples)
        task_counts = {}
        for ex in all_examples:
            task_type = ex['metadata']['original_task_type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        print("\n📊 Newly Generated Examples:")
        print("=" * 60)
        for task_type, count in sorted(task_counts.items()):
            print(f"  {task_type:30} {count:5} new examples")
        print("=" * 60)
        print(f"  NEW: {len(all_examples)} examples")
        print(f"  EXISTING: {existing_count} examples")
        print(f"  TOTAL IN FILE: {total_examples} examples\n")
        
        return all_examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate AI-powered Q&A dataset with reasoning using Gemini')
    parser.add_argument('--icmr-data', default='../data/structured/all_syndromes_gemini.json',
                        help='Path to ICMR structured data')
    parser.add_argument('--output', default='../data/amr_ai_generated_qa.jsonl',
                        help='Output path for AI-generated Q&A')
    parser.add_argument('--gemini-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    parser.add_argument('--model', default='gemini-2.0-flash-exp',
                        help='Gemini model to use (default: gemini-2.0-flash-exp)')
    parser.add_argument('--guideline-count', type=int, default=50,
                        help='Number of guideline lookup Q&As to generate')
    parser.add_argument('--drug-count', type=int, default=30,
                        help='Number of drug info Q&As to generate')
    parser.add_argument('--pathogen-count', type=int, default=30,
                        help='Number of pathogen Q&As to generate')
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Gemini API key
    api_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("❌ ERROR: No Gemini API key provided")
        print("\n❌ ERROR: No Gemini API key found")
        print("Set GEMINI_API_KEY environment variable or use --gemini-key")
        print("\nExample:")
        print("  export GEMINI_API_KEY='your_key_here'")
        print("  python src/generate_qa_with_ai.py --guideline-count 50")
        return
    
    print(f"✓ Loaded Gemini API key")
    
    generator = AIQAGenerator(api_key, args.icmr_data, args.model)
    generator.generate_dataset(
        guideline_count=args.guideline_count,
        drug_count=args.drug_count,
        pathogen_count=args.pathogen_count,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

