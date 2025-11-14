"""
Reprocess failed syndrome extractions.

This script identifies syndromes with missing pathogen/drug data and re-extracts them
using improved methods. It preserves successful extractions and only updates failures.
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_extractor import ICMRStructuredExtractor
from src.data_models import SyndromeData
from src.utils import load_groq_api_keys, setup_logging

logger = logging.getLogger(__name__)


class SyndromeReprocessor:
    """Reprocess failed syndrome extractions."""
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize reprocessor.
        
        Args:
            api_keys: List of Groq API keys
        """
        self.api_keys = api_keys
        self.extractor = ICMRStructuredExtractor(
            api_keys=api_keys,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
        self.reprocessing_log = []
    
    def load_existing_results(self, input_file: str) -> List[Dict]:
        """
        Load existing syndrome extraction results.
        
        Args:
            input_file: Path to all_syndromes.json
            
        Returns:
            List of syndrome dictionaries
        """
        logger.info(f"Loading existing results from {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                syndromes = json.load(f)
            
            logger.info(f"Loaded {len(syndromes)} existing syndromes")
            return syndromes
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
            raise
    
    def identify_failed_syndromes(self, syndromes: List[Dict]) -> Tuple[List[int], Dict[str, str]]:
        """
        Identify syndromes that need reprocessing.
        
        Failure criteria:
        - Missing pathogens when syndrome should have them
        - Missing first-line drugs
        - Definition is placeholder text
        
        Args:
            syndromes: List of syndrome dictionaries
            
        Returns:
            Tuple of (list of indices to reprocess, dict of failure reasons)
        """
        logger.info("Identifying failed syndromes...")
        
        failed_indices = []
        failure_reasons = {}
        
        # Keywords that indicate syndrome should have pathogens
        pathogen_expected_keywords = [
            'infection', 'syndrome', 'fever', 'sepsis', 'pneumonia',
            'meningitis', 'encephalitis', 'abscess', 'cellulitis',
            'pyelonephritis', 'cystitis', 'gastroenteritis'
        ]
        
        for i, syndrome in enumerate(syndromes):
            reasons = []
            syndrome_name = syndrome.get('syndrome_name', '').lower()
            
            # Check if pathogens are missing when expected
            common_pathogens = syndrome.get('common_pathogens', [])
            should_have_pathogens = any(
                keyword in syndrome_name for keyword in pathogen_expected_keywords
            )
            
            if should_have_pathogens and len(common_pathogens) == 0:
                reasons.append("Missing pathogens (expected for this syndrome type)")
            
            # Check if first-line drugs are missing
            first_line_drugs = syndrome.get('first_line_drugs', [])
            if len(first_line_drugs) == 0:
                reasons.append("Missing first-line drugs")
            
            # Check if definition is placeholder
            definition = syndrome.get('definition', '')
            if 'not provided' in definition.lower() or len(definition) < 50:
                reasons.append("Insufficient definition")
            
            # Check if syndrome name is generic
            if syndrome_name in ['unnamed syndrome', 'syndrome']:
                reasons.append("Generic syndrome name")
            
            # If any failure criteria met, mark for reprocessing
            if reasons:
                failed_indices.append(i)
                failure_reasons[i] = '; '.join(reasons)
        
        logger.info(f"Found {len(failed_indices)} syndromes needing reprocessing")
        
        # Log details
        for idx in failed_indices[:10]:  # Show first 10
            syndrome_name = syndromes[idx].get('syndrome_name', 'Unknown')
            logger.info(f"  - [{idx}] {syndrome_name}: {failure_reasons[idx]}")
        
        if len(failed_indices) > 10:
            logger.info(f"  ... and {len(failed_indices) - 10} more")
        
        return failed_indices, failure_reasons
    
    def load_raw_text(self, page_start: int, page_end: int, extracted_dir: str = 'data/extracted') -> Tuple[str, List[Dict]]:
        """
        Load raw extracted text and tables for a syndrome.
        
        Args:
            page_start: Starting page number
            page_end: Ending page number
            extracted_dir: Directory containing extracted page files
            
        Returns:
            Tuple of (combined text, list of tables)
        """
        text_parts = []
        all_tables = []
        
        extracted_path = Path(extracted_dir)
        
        for page_num in range(page_start, page_end + 1):
            # Load text
            text_file = extracted_path / f"page_{page_num:03d}_text.json"
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    text_parts.append(page_data.get('text', ''))
            
            # Load tables (if any)
            table_num = 0
            while True:
                table_file = extracted_path / f"page_{page_num:03d}_table_{table_num}.csv"
                if not table_file.exists():
                    break
                
                try:
                    df = pd.read_csv(table_file)
                    table_dict = {
                        'headers': df.columns.tolist(),
                        'rows': df.values.tolist(),
                        'table_number': table_num
                    }
                    all_tables.append(table_dict)
                    table_num += 1
                except Exception as e:
                    logger.warning(f"Failed to load table {table_file}: {e}")
                    break
        
        combined_text = '\n\n'.join(text_parts)
        return combined_text, all_tables
    
    def reprocess_syndrome(
        self,
        syndrome_data: Dict,
        index: int,
        failure_reason: str
    ) -> Tuple[Optional[Dict], str]:
        """
        Reprocess a single failed syndrome.
        
        Args:
            syndrome_data: Original syndrome data
            index: Index in the original list
            failure_reason: Why this syndrome failed
            
        Returns:
            Tuple of (new syndrome data dict or None, status message)
        """
        syndrome_name = syndrome_data.get('syndrome_name', 'Unknown')
        page_start = syndrome_data.get('source_page', 0)
        page_end = page_start + 5  # Assume 5-page sections
        source_pdf = syndrome_data.get('source_document', 'unknown')
        
        logger.info(f"Reprocessing [{index}] {syndrome_name}")
        logger.info(f"  Reason: {failure_reason}")
        logger.info(f"  Pages: {page_start}-{page_end}")
        
        try:
            # Load raw text
            raw_text, tables = self.load_raw_text(page_start, page_end)
            
            if not raw_text or len(raw_text) < 100:
                logger.warning(f"  Insufficient raw text for {syndrome_name}")
                return None, "Insufficient raw text"
            
            # Re-extract using LLM
            new_syndrome_data = self.extractor.extract_syndrome_data(
                syndrome_text=raw_text,
                tables_list=tables,
                page_number=page_start,
                source_pdf=source_pdf
            )
            
            if not new_syndrome_data:
                logger.warning(f"  Re-extraction failed for {syndrome_name}")
                return None, "Re-extraction failed"
            
            # Calculate quality improvement
            old_pathogen_count = len(syndrome_data.get('common_pathogens', []))
            new_pathogen_count = len(new_syndrome_data.common_pathogens)
            
            old_drug_count = len(syndrome_data.get('first_line_drugs', []))
            new_drug_count = len(new_syndrome_data.first_line_drugs)
            
            improvement = {
                'pathogens': new_pathogen_count - old_pathogen_count,
                'drugs': new_drug_count - old_drug_count
            }
            
            # Convert to dict
            new_data_dict = new_syndrome_data.model_dump()
            
            # Add reprocessing metadata
            new_data_dict['reprocessed'] = True
            new_data_dict['reprocessed_at'] = datetime.now().isoformat()
            new_data_dict['original_failure'] = failure_reason
            
            status = f"Success: +{improvement['pathogens']} pathogens, +{improvement['drugs']} drugs"
            logger.info(f"  ✓ {status}")
            
            return new_data_dict, status
            
        except Exception as e:
            logger.error(f"  Error reprocessing {syndrome_name}: {e}")
            return None, f"Error: {str(e)}"
    
    def merge_results(
        self,
        original_syndromes: List[Dict],
        reprocessed_data: Dict[int, Dict]
    ) -> List[Dict]:
        """
        Merge original and reprocessed results.
        
        Args:
            original_syndromes: Original syndrome list
            reprocessed_data: Dict of {index: new_syndrome_data}
            
        Returns:
            Merged syndrome list
        """
        logger.info("Merging original and reprocessed results...")
        
        merged = []
        for i, syndrome in enumerate(original_syndromes):
            if i in reprocessed_data:
                # Use reprocessed version
                merged.append(reprocessed_data[i])
                logger.debug(f"  Replaced [{i}] {syndrome.get('syndrome_name')}")
            else:
                # Keep original
                merged.append(syndrome)
        
        logger.info(f"Merged {len(merged)} syndromes ({len(reprocessed_data)} reprocessed)")
        return merged
    
    def generate_comparison_report(
        self,
        original_syndromes: List[Dict],
        merged_syndromes: List[Dict],
        reprocessed_indices: List[int],
        output_dir: str
    ) -> None:
        """
        Generate before/after comparison report.
        
        Args:
            original_syndromes: Original syndrome list
            merged_syndromes: Merged syndrome list
            reprocessed_indices: Indices that were reprocessed
            output_dir: Directory to save report
        """
        logger.info("Generating comparison report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SYNDROME REPROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Calculate statistics
        def calc_stats(syndromes):
            total = len(syndromes)
            missing_pathogens = sum(1 for s in syndromes if len(s.get('common_pathogens', [])) == 0)
            missing_drugs = sum(1 for s in syndromes if len(s.get('first_line_drugs', [])) == 0)
            total_pathogens = sum(len(s.get('common_pathogens', [])) for s in syndromes)
            total_drugs = sum(len(s.get('first_line_drugs', [])) for s in syndromes)
            return {
                'total': total,
                'missing_pathogens': missing_pathogens,
                'missing_drugs': missing_drugs,
                'total_pathogens': total_pathogens,
                'total_drugs': total_drugs,
                'valid': total - missing_pathogens - missing_drugs
            }
        
        before_stats = calc_stats(original_syndromes)
        after_stats = calc_stats(merged_syndromes)
        
        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total syndromes: {before_stats['total']}")
        report_lines.append(f"Syndromes reprocessed: {len(reprocessed_indices)}")
        report_lines.append("")
        
        # Before/After comparison
        report_lines.append("BEFORE REPROCESSING:")
        report_lines.append(f"  Missing pathogens: {before_stats['missing_pathogens']}")
        report_lines.append(f"  Missing drugs: {before_stats['missing_drugs']}")
        report_lines.append(f"  Total pathogens: {before_stats['total_pathogens']}")
        report_lines.append(f"  Total drugs: {before_stats['total_drugs']}")
        report_lines.append("")
        
        report_lines.append("AFTER REPROCESSING:")
        report_lines.append(f"  Missing pathogens: {after_stats['missing_pathogens']}")
        report_lines.append(f"  Missing drugs: {after_stats['missing_drugs']}")
        report_lines.append(f"  Total pathogens: {after_stats['total_pathogens']}")
        report_lines.append(f"  Total drugs: {after_stats['total_drugs']}")
        report_lines.append("")
        
        # Improvements
        report_lines.append("IMPROVEMENTS:")
        pathogen_improvement = before_stats['missing_pathogens'] - after_stats['missing_pathogens']
        drug_improvement = before_stats['missing_drugs'] - after_stats['missing_drugs']
        pathogen_total_improvement = after_stats['total_pathogens'] - before_stats['total_pathogens']
        drug_total_improvement = after_stats['total_drugs'] - before_stats['total_drugs']
        
        report_lines.append(f"  Syndromes with pathogens fixed: {pathogen_improvement}")
        report_lines.append(f"  Syndromes with drugs fixed: {drug_improvement}")
        report_lines.append(f"  Additional pathogens extracted: {pathogen_total_improvement}")
        report_lines.append(f"  Additional drugs extracted: {drug_total_improvement}")
        report_lines.append("")
        
        # Still failing
        still_failing = [
            i for i in reprocessed_indices
            if len(merged_syndromes[i].get('common_pathogens', [])) == 0 or
               len(merged_syndromes[i].get('first_line_drugs', [])) == 0
        ]
        
        if still_failing:
            report_lines.append("SYNDROMES STILL NEEDING ATTENTION:")
            report_lines.append(f"  {len(still_failing)} syndromes still have issues")
            for idx in still_failing[:10]:
                syndrome = merged_syndromes[idx]
                issues = []
                if len(syndrome.get('common_pathogens', [])) == 0:
                    issues.append("No pathogens")
                if len(syndrome.get('first_line_drugs', [])) == 0:
                    issues.append("No drugs")
                report_lines.append(f"  - [{idx}] {syndrome.get('syndrome_name')}: {', '.join(issues)}")
        else:
            report_lines.append("✓ ALL SYNDROMES SUCCESSFULLY FIXED!")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_file = output_path / "reprocessing_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved comparison report to {report_file}")
        
        # Print to console
        print('\n'.join(report_lines))
    
    def create_manual_review_queue(
        self,
        syndromes: List[Dict],
        failed_indices: List[int],
        output_dir: str
    ) -> None:
        """
        Create CSV for manual review of still-failed syndromes.
        
        Args:
            syndromes: Merged syndrome list
            failed_indices: Indices still failing after reprocessing
            output_dir: Directory to save CSV
        """
        if not failed_indices:
            logger.info("No syndromes need manual review - all fixed!")
            return
        
        logger.info(f"Creating manual review queue for {len(failed_indices)} syndromes...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        review_data = []
        for idx in failed_indices:
            syndrome = syndromes[idx]
            
            issues = []
            if len(syndrome.get('common_pathogens', [])) == 0:
                issues.append("Missing pathogens")
            if len(syndrome.get('first_line_drugs', [])) == 0:
                issues.append("Missing drugs")
            
            # Get text excerpt
            definition = syndrome.get('definition', '')
            excerpt = definition[:200] + "..." if len(definition) > 200 else definition
            
            review_data.append({
                'index': idx,
                'syndrome_name': syndrome.get('syndrome_name', ''),
                'page': syndrome.get('source_page', ''),
                'issues': '; '.join(issues),
                'text_excerpt': excerpt,
                'pathogens_found': len(syndrome.get('common_pathogens', [])),
                'drugs_found': len(syndrome.get('first_line_drugs', []))
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(review_data)
        csv_file = output_path / "manual_review_queue.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Saved manual review queue to {csv_file}")


def main():
    """Main reprocessing function."""
    parser = argparse.ArgumentParser(
        description='Reprocess failed syndrome extractions'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/structured/all_syndromes.json',
        help='Input JSON file with existing results'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/structured/all_syndromes_v2.json',
        help='Output JSON file for merged results'
    )
    
    parser.add_argument(
        '--extracted-dir',
        type=str,
        default='data/extracted',
        help='Directory containing extracted page text/tables'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/structured',
        help='Directory for output files and reports'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if syndrome looks OK'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("=" * 80)
    print("SYNDROME REPROCESSING TOOL")
    print("=" * 80)
    print()
    
    # Load API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = load_groq_api_keys()
    if not api_keys:
        print("❌ ERROR: No Groq API keys found")
        print("Please ensure .env file has GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")
        return 1
    
    print(f"✓ Loaded {len(api_keys)} API key(s)")
    print()
    
    # Initialize reprocessor
    reprocessor = SyndromeReprocessor(api_keys)
    
    # Step 1: Load existing results
    print("Step 1: Loading existing results...")
    print("-" * 80)
    try:
        original_syndromes = reprocessor.load_existing_results(args.input)
        print(f"✓ Loaded {len(original_syndromes)} syndromes")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return 1
    
    print()
    
    # Step 2: Identify failures
    print("Step 2: Identifying failed syndromes...")
    print("-" * 80)
    failed_indices, failure_reasons = reprocessor.identify_failed_syndromes(original_syndromes)
    print(f"✓ Found {len(failed_indices)} syndromes needing reprocessing")
    
    if not failed_indices:
        print("✓ All syndromes look good! No reprocessing needed.")
        return 0
    
    print()
    
    # Step 3: Reprocess failures
    print("Step 3: Reprocessing failed syndromes...")
    print("-" * 80)
    
    reprocessed_data = {}
    still_failed = []
    
    from tqdm import tqdm
    for idx in tqdm(failed_indices, desc="Reprocessing"):
        syndrome = original_syndromes[idx]
        new_data, status = reprocessor.reprocess_syndrome(
            syndrome_data=syndrome,
            index=idx,
            failure_reason=failure_reasons[idx]
        )
        
        if new_data:
            reprocessed_data[idx] = new_data
        else:
            still_failed.append(idx)
    
    print(f"\n✓ Reprocessed {len(reprocessed_data)} syndromes")
    if still_failed:
        print(f"⚠️  {len(still_failed)} syndromes still failed")
    
    print()
    
    # Step 4: Merge results
    print("Step 4: Merging results...")
    print("-" * 80)
    merged_syndromes = reprocessor.merge_results(original_syndromes, reprocessed_data)
    print(f"✓ Created merged dataset with {len(merged_syndromes)} syndromes")
    
    # Save merged results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged_syndromes, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {args.output}")
    
    print()
    
    # Step 5: Generate reports
    print("Step 5: Generating reports...")
    print("-" * 80)
    reprocessor.generate_comparison_report(
        original_syndromes=original_syndromes,
        merged_syndromes=merged_syndromes,
        reprocessed_indices=list(reprocessed_data.keys()),
        output_dir=args.output_dir
    )
    
    # Create manual review queue if needed
    if still_failed:
        reprocessor.create_manual_review_queue(
            syndromes=merged_syndromes,
            failed_indices=still_failed,
            output_dir=args.output_dir
        )
    
    print()
    print("=" * 80)
    print("REPROCESSING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review: data/structured/reprocessing_report.txt")
    print("2. Check merged results: data/structured/all_syndromes_v2.json")
    if still_failed:
        print("3. Manual review needed: data/structured/manual_review_queue.csv")
    else:
        print("3. ✓ All syndromes successfully fixed!")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

