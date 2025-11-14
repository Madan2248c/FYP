"""
Multi-Agent Reasoning Refiner - Main Coordinator

Orchestrates Generator and Verifier agents to produce high-quality reasoning traces
for prescription validation. Implements iterative refinement with up to 2 retries.

Usage:
    python src/multi_agent_refiner.py --input data/patient_cases.json --output data/reasoning_multi_agent/
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import csv
from tqdm import tqdm
from difflib import SequenceMatcher

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.generator_agent import GeneratorAgent, GeneratorOutput
from src.agents.verifier_agent import VerifierAgent, VerificationResult

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """
    Coordinates Generator and Verifier agents for reasoning refinement.
    
    Workflow:
    1. Generator creates reasoning
    2. Verifier validates reasoning
    3. If invalid, regenerate (max 2 retries)
    4. Save validated reasoning
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        reference_dataset_path: str,
        max_retries: int = 2,
        generator_model: str = "gemini-2.5-pro",
        thinking_budget: int = -1,
        dosage_tolerance: float = 0.10
    ):
        """
        Initialize Multi-Agent Coordinator.
        
        Args:
            gemini_api_key: API key for Gemini (Generator uses this)
            reference_dataset_path: Path to all_syndromes_gemini.json
            max_retries: Maximum regeneration attempts
            generator_model: Model for Generator agent (default: gemini-2.5-pro)
            thinking_budget: Thinking token budget (-1 for dynamic, 128-32768 for fixed)
                            See: https://ai.google.dev/gemini-api/docs/thinking
            dosage_tolerance: Allowable dosage deviation for Verifier
        """
        self.max_retries = max_retries
        
        # Initialize agents
        self.generator = GeneratorAgent(
            api_key=gemini_api_key,
            model_name=generator_model,
            thinking_budget=thinking_budget
        )
        self.verifier = VerifierAgent(dosage_tolerance=dosage_tolerance)
        
        # Load ICMR reference data
        self.reference_data = self._load_reference_data(reference_dataset_path)
        
        # Statistics
        self.stats = {
            'total_cases': 0,
            'successful': 0,
            'failed': 0,
            'retries_needed': 0,
            'total_retries': 0
        }
        
        logger.info(f"Multi-Agent Coordinator initialized")
        logger.info(f"Loaded {len(self.reference_data)} syndromes from reference dataset")
    
    def _load_reference_data(self, path: str) -> Dict[str, Dict]:
        """Load ICMR reference data indexed by syndrome name."""
        logger.info(f"Loading reference data from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            syndromes = json.load(f)
        
        # Index by syndrome name for fast lookup
        indexed = {}
        for syndrome in syndromes:
            name = syndrome.get('syndrome_name', '').lower()
            indexed[name] = syndrome
        
        return indexed
    
    def process_case(
        self,
        case: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process a single case with Generator-Verifier loop.
        
        Args:
            case: Patient case dictionary
            
        Returns:
            Tuple of (validated_reasoning, metadata)
        """
        case_id = case.get('case_id', 'UNKNOWN')
        diagnosis = case.get('diagnosis', '').lower()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing case: {case_id}")
        logger.info(f"Diagnosis: {diagnosis}")
        
        # Find matching guideline
        guideline = self._find_matching_guideline(diagnosis)
        
        if not guideline:
            logger.warning(f"No matching guideline found for '{diagnosis}'")
            return None, {
                'case_id': case_id,
                'status': 'no_guideline',
                'retries': 0,
                'error': f"No guideline match for '{diagnosis}'",
                'validation_checks': 0,
                'warnings': ''
            }
        
        # Try generation with retries
        verification_result = None  # Initialize to avoid UnboundLocalError
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{self.max_retries}")
                self.stats['total_retries'] += 1
            
            # Generate reasoning
            reasoning_output = self.generator.generate_reasoning(case, guideline)
            
            if not reasoning_output:
                logger.error(f"Generator failed for case {case_id}")
                continue
            
            # Convert to dict for verification
            reasoning_dict = self._pydantic_to_dict(reasoning_output)
            
            # Verify reasoning
            verification_result = self.verifier.verify_reasoning(
                reasoning_dict,
                case,
                guideline
            )
            
            if verification_result.valid:
                logger.info(f"✓ Reasoning validated on attempt {attempt + 1}")
                
                # Prepare final output
                validated_reasoning = self._prepare_final_output(
                    case,
                    reasoning_dict,
                    verification_result,
                    guideline
                )
                
                metadata = {
                    'case_id': case_id,
                    'status': 'success',
                    'retries': attempt,
                    'validation_checks': verification_result.checks_performed,
                    'warnings': ', '.join(verification_result.warnings) if verification_result.warnings else '',
                    'error': ''
                }
                
                self.stats['successful'] += 1
                if attempt > 0:
                    self.stats['retries_needed'] += 1
                
                return validated_reasoning, metadata
            else:
                logger.warning(f"✗ Validation failed: {verification_result.errors}")
                
                if attempt < self.max_retries:
                    logger.info(f"Will retry generation...")
                else:
                    logger.error(f"Max retries reached for case {case_id}")
        
        # Failed after all retries
        self.stats['failed'] += 1
        last_errors = verification_result.errors if verification_result else []
        return None, {
            'case_id': case_id,
            'status': 'failed',
            'retries': self.max_retries,
            'error': '; '.join(last_errors) if last_errors else 'Validation failed after max retries',
            'validation_checks': verification_result.checks_performed if verification_result else 0,
            'warnings': ''
        }
    
    def process_batch(
        self,
        cases: List[Dict[str, Any]],
        output_dir: str,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process a batch of cases and save results.
        Saves progress incrementally and supports resume.
        
        Args:
            cases: List of patient cases
            output_dir: Directory to save outputs
            resume: If True, skip already processed cases (default: True)
            
        Returns:
            Summary statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Output files
        validated_file = output_path / "reasoning_dataset_validated.jsonl"
        log_file = output_path / "reasoning_log.csv"
        metrics_file = output_path / "metrics.json"
        
        self.stats['total_cases'] = len(cases)
        
        # Load existing results if resuming
        processed_case_ids = set()
        if resume and validated_file.exists():
            logger.info(f"📂 Resume mode: Loading existing results from {validated_file}")
            with open(validated_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            case_id = data.get('context', {}).get('patient_profile', {}).get('case_id')
                            if not case_id:
                                case_id = data.get('case_id')
                            if case_id:
                                processed_case_ids.add(case_id)
                        except:
                            pass
            logger.info(f"✓ Found {len(processed_case_ids)} already processed cases - will skip them\n")
        
        validated_outputs = []
        log_entries = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(cases)} cases")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*60}\n")
        
        # Process each case
        for case in tqdm(cases, desc="Processing cases"):
            case_id = case.get('case_id', 'UNKNOWN')
            
            # Skip if already processed
            if case_id in processed_case_ids:
                logger.info(f"⏭️  Skipping {case_id} (already processed)")
                self.stats['total_cases'] -= 1  # Don't count skipped cases
                continue
            
            reasoning, metadata = self.process_case(case)
            
            if reasoning:
                validated_outputs.append(reasoning)
                
                # Save immediately (incremental save)
                with open(validated_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(reasoning, ensure_ascii=False) + '\n')
                logger.info(f"💾 Saved {case_id} to disk")
            
            log_entries.append(metadata)
            
            # Update CSV log incrementally
            self._append_to_csv_log(log_file, metadata)
        
        # Don't rewrite the file - already saved incrementally
        logger.info(f"\n✓ Saved {len(validated_outputs)} validated reasoning traces to {validated_file}")
        
        # Save log
        fieldnames = ['case_id', 'status', 'retries', 'validation_checks', 'warnings', 'error']
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if log_entries:
                writer.writerows(log_entries)
        
        logger.info(f"✓ Saved processing log to {log_file}")
        
        # Calculate and save metrics
        metrics = self._calculate_metrics(validated_outputs, log_entries)
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Saved metrics to {metrics_file}")
        
        # Print summary
        self._print_summary(metrics)
        
        return metrics
    
    def _find_matching_guideline(self, diagnosis: str) -> Dict:
        """
        Find matching ICMR guideline for diagnosis using fuzzy matching.
        
        Matching strategy:
        1. Exact match
        2. Partial match (substring)
        3. Fuzzy match (75%+ similarity after cleaning)
        """
        diagnosis_lower = diagnosis.lower().strip()
        
        # Try exact match first
        if diagnosis_lower in self.reference_data:
            logger.debug(f"Exact match: '{diagnosis}' → '{diagnosis_lower}'")
            return self.reference_data[diagnosis_lower]
        
        # Try partial match (substring)
        for syndrome_name, guideline in self.reference_data.items():
            if diagnosis_lower in syndrome_name or syndrome_name in diagnosis_lower:
                logger.info(f"Partial match: '{diagnosis}' → '{syndrome_name}'")
                return guideline
        
        # Try fuzzy match with cleaned diagnosis
        diagnosis_clean = self._clean_diagnosis(diagnosis_lower)
        
        best_match = None
        best_score = 0
        best_syndrome = None
        
        for syndrome_name, guideline in self.reference_data.items():
            syndrome_clean = self._clean_diagnosis(syndrome_name)
            
            # Calculate similarity ratio
            score = SequenceMatcher(None, diagnosis_clean, syndrome_clean).ratio()
            
            if score > best_score:
                best_score = score
                best_match = guideline
                best_syndrome = syndrome_name
        
        # Accept match if similarity >= 75%
        if best_score >= 0.75:
            logger.info(f"Fuzzy match: '{diagnosis}' → '{best_syndrome}' (similarity: {best_score:.2%})")
            return best_match
        
        logger.debug(f"No match found for '{diagnosis}' (best: '{best_syndrome}' at {best_score:.2%})")
        return None
    
    def _clean_diagnosis(self, text: str) -> str:
        """
        Clean diagnosis by removing common qualifiers.
        
        This helps match "bacterial meningitis" to "meningitis syndrome"
        or "complicated UTI" to "UTI".
        """
        qualifiers = [
            'acute', 'chronic', 'complicated', 'uncomplicated',
            'bacterial', 'viral', 'fungal', 'parasitic',
            'severe', 'moderate', 'mild',
            'primary', 'secondary',
            'community', 'hospital', 'healthcare',
            'acquired', 'associated',
            'syndrome', 'disease', 'infection', 'infections'
        ]
        
        words = text.lower().split()
        clean_words = [w for w in words if w not in qualifiers]
        
        return ' '.join(clean_words) if clean_words else text
    
    def _append_to_csv_log(self, log_file: Path, metadata: Dict):
        """Append metadata to CSV log file (create if doesn't exist)."""
        fieldnames = ['case_id', 'status', 'retries', 'validation_checks', 'warnings', 'error']
        
        file_exists = log_file.exists()
        
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metadata)
    
    def _pydantic_to_dict(self, pydantic_obj: GeneratorOutput) -> Dict:
        """Convert Pydantic model to dictionary."""
        return {
            'case_id': pydantic_obj.case_id,
            'reasoning_steps': [
                {
                    'step_number': step.step_number,
                    'description': step.description,
                    'observation': step.observation,
                    'reference': step.reference
                }
                for step in pydantic_obj.reasoning_steps
            ],
            'final_decision': pydantic_obj.final_decision,
            'justification': pydantic_obj.justification,
            'icmr_references': pydantic_obj.icmr_references,
            'confidence_score': pydantic_obj.confidence_score
        }
    
    def _prepare_final_output(
        self,
        case: Dict,
        reasoning: Dict,
        verification: VerificationResult,
        guideline: Dict
    ) -> Dict:
        """Prepare final output in fine-tuning format."""
        return {
            'instruction': "Validate prescription according to ICMR 2025 Antimicrobial Treatment Guidelines.",
            'context': {
                'patient_profile': case.get('patient_profile', {}),
                'diagnosis': case.get('diagnosis'),
                'symptoms': case.get('symptoms', []),
                'prescription': case.get('prescription', {})
            },
            'generator_reasoning_steps': [
                f"Step {step['step_number']}: {step['description']} - {step['observation']}"
                for step in reasoning['reasoning_steps']
            ],
            'final_answer': f"{reasoning['final_decision']}: {reasoning['justification']}",
            'verifier_result': verification.to_dict()['summary'],
            'icmr_reference': reasoning['icmr_references'][0] if reasoning['icmr_references'] else f"ICMR 2025, {guideline.get('syndrome_name')}, Page {guideline.get('source_page')}",
            'confidence_score': reasoning.get('confidence_score', 0.0),
            'validation_checks': verification.checks_performed,
            'case_id': case.get('case_id'),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_metrics(
        self,
        validated_outputs: List[Dict],
        log_entries: List[Dict]
    ) -> Dict:
        """Calculate quality metrics."""
        total_cases = self.stats['total_cases']
        successful = self.stats['successful']
        failed = self.stats['failed']
        
        metrics = {
            'summary': {
                'total_cases': total_cases,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_cases if total_cases > 0 else 0,
                'retries_needed': self.stats['retries_needed'],
                'avg_retries': self.stats['total_retries'] / total_cases if total_cases > 0 else 0
            },
            'reasoning_quality': {},
            'validation_stats': {}
        }
        
        if validated_outputs:
            # Average reasoning length
            avg_steps = sum(len(o['generator_reasoning_steps']) for o in validated_outputs) / len(validated_outputs)
            metrics['reasoning_quality']['avg_reasoning_steps'] = avg_steps
            
            # Confidence scores
            confidence_scores = [o.get('confidence_score', 0) for o in validated_outputs]
            metrics['reasoning_quality']['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
            metrics['reasoning_quality']['min_confidence'] = min(confidence_scores)
            metrics['reasoning_quality']['max_confidence'] = max(confidence_scores)
            
            # Valid reasoning rate (passed without regeneration)
            first_pass = sum(1 for e in log_entries if e.get('status') == 'success' and e.get('retries', 0) == 0)
            metrics['validation_stats']['first_pass_rate'] = first_pass / successful if successful > 0 else 0
        
        return metrics
    
    def _print_summary(self, metrics: Dict):
        """Print summary statistics."""
        summary = metrics['summary']
        quality = metrics.get('reasoning_quality', {})
        validation = metrics.get('validation_stats', {})
        
        print(f"\n{'='*60}")
        print("MULTI-AGENT REASONING GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"\n📊 Overall Statistics:")
        print(f"  Total cases: {summary['total_cases']}")
        print(f"  Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)")
        print(f"  Failed: {summary['failed']}")
        print(f"  Cases needing retries: {summary['retries_needed']}")
        print(f"  Average retries per case: {summary['avg_retries']:.2f}")
        
        if quality:
            print(f"\n🎯 Reasoning Quality:")
            print(f"  Average reasoning steps: {quality.get('avg_reasoning_steps', 0):.1f}")
            print(f"  Average confidence: {quality.get('avg_confidence', 0):.2f}")
            print(f"  Confidence range: {quality.get('min_confidence', 0):.2f} - {quality.get('max_confidence', 0):.2f}")
        
        if validation:
            print(f"\n✅ Validation Statistics:")
            print(f"  First-pass success rate: {validation.get('first_pass_rate', 0)*100:.1f}%")
        
        print(f"\n{'='*60}\n")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent Reasoning Refiner for Prescription Validation'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/patient_cases.json',
        help='Input JSON file with patient cases'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/reasoning_multi_agent/',
        help='Output directory for validated reasoning'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        default='data/structured/all_syndromes_gemini.json',
        help='Path to ICMR reference dataset'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Maximum regeneration attempts per case'
    )
    
    parser.add_argument(
        '--generator-model',
        type=str,
        default='gemini-2.5-pro',
        help='Model for Generator agent (default: gemini-2.5-pro with thinking)'
    )
    
    parser.add_argument(
        '--thinking-budget',
        type=int,
        default=-1,
        help='Thinking token budget: -1 for dynamic (default), 0 to disable, 128-32768 for fixed budget'
    )
    
    parser.add_argument(
        '--gemini-key',
        type=str,
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Get API key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("❌ ERROR: No Gemini API key provided")
        print("Set GEMINI_API_KEY environment variable or use --gemini-key")
        return 1
    
    # Check input file
    if not Path(args.input).exists():
        print(f"❌ ERROR: Input file not found: {args.input}")
        return 1
    
    if not Path(args.reference).exists():
        print(f"❌ ERROR: Reference dataset not found: {args.reference}")
        return 1
    
    print(f"\n{'='*60}")
    print("MULTI-AGENT REASONING REFINER")
    print(f"{'='*60}\n")
    
    # Load cases
    with open(args.input, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    print(f"✓ Loaded {len(cases)} patient cases from {args.input}")
    print(f"✓ Reference dataset: {args.reference}")
    print(f"✓ Output directory: {args.output}")
    print(f"✓ Generator model: {args.generator_model}")
    print(f"✓ Thinking budget: {'Dynamic' if args.thinking_budget == -1 else f'{args.thinking_budget} tokens'}")
    print(f"✓ Max retries: {args.max_retries}\n")
    
    # Initialize coordinator
    coordinator = MultiAgentCoordinator(
        gemini_api_key=gemini_key,
        reference_dataset_path=args.reference,
        max_retries=args.max_retries,
        generator_model=args.generator_model,
        thinking_budget=args.thinking_budget
    )
    
    # Process batch
    metrics = coordinator.process_batch(cases, args.output)
    
    print(f"\n✅ Processing complete!")
    print(f"\nOutputs saved to: {args.output}")
    print(f"  - reasoning_dataset_validated.jsonl (validated reasoning traces)")
    print(f"  - reasoning_log.csv (processing log)")
    print(f"  - metrics.json (quality metrics)\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

