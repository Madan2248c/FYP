"""
Reprocess failed syndromes using Gemini's native PDF processing.

Instead of sending extracted text, this sends the PDF directly to Gemini,
allowing it to see tables, formatting, and full context for better extraction.
"""

import json
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from google import genai
from google.genai import types

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_models import SyndromeData
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class GeminiPDFReprocessor:
    """Reprocess failed syndromes using Gemini's native PDF processing."""
    
    def __init__(self, gemini_api_key: str, pdf_path: str):
        """
        Initialize Gemini reprocessor.
        
        Args:
            gemini_api_key: Gemini API key
            pdf_path: Path to ICMR PDF
        """
        self.gemini_api_key = gemini_api_key
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Initialize Gemini client (correct API)
        self.client = genai.Client(api_key=gemini_api_key)
        
        # Read PDF bytes
        logger.info(f"Loading PDF: {pdf_path}")
        self.pdf_bytes = self.pdf_path.read_bytes()
        
        logger.info("✓ Gemini client initialized")
    
    def load_existing_results(self, input_file: str) -> List[Dict]:
        """Load existing syndrome extraction results."""
        logger.info(f"Loading existing results from {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                syndromes = json.load(f)
            
            logger.info(f"Loaded {len(syndromes)} existing syndromes")
            return syndromes
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
            raise
    
    def identify_failed_syndromes(self, syndromes: List[Dict]) -> List[int]:
        """
        Identify syndromes that need reprocessing.
        
        Args:
            syndromes: List of syndrome dictionaries
            
        Returns:
            List of indices to reprocess
        """
        logger.info("Identifying failed syndromes...")
        
        failed_indices = []
        
        pathogen_expected_keywords = [
            'infection', 'syndrome', 'fever', 'sepsis', 'pneumonia',
            'meningitis', 'encephalitis', 'abscess', 'cellulitis',
            'pyelonephritis', 'cystitis', 'gastroenteritis'
        ]
        
        for i, syndrome in enumerate(syndromes):
            syndrome_name = syndrome.get('syndrome_name', '').lower()
            common_pathogens = syndrome.get('common_pathogens', [])
            first_line_drugs = syndrome.get('first_line_drugs', [])
            
            should_have_pathogens = any(
                keyword in syndrome_name for keyword in pathogen_expected_keywords
            )
            
            # Mark for reprocessing if missing critical data
            if (should_have_pathogens and len(common_pathogens) == 0) or len(first_line_drugs) == 0:
                failed_indices.append(i)
        
        logger.info(f"Found {len(failed_indices)} syndromes needing reprocessing")
        return failed_indices
    
    def extract_with_gemini(
        self,
        syndrome_name: str,
        page_start: int,
        page_end: int
    ) -> Optional[Dict]:
        """
        Extract syndrome data using Gemini's native PDF processing.
        
        Args:
            syndrome_name: Name of syndrome
            page_start: Starting page number
            page_end: Ending page number
            
        Returns:
            Extracted syndrome data or None
        """
        logger.info(f"Extracting {syndrome_name} (pages {page_start}-{page_end}) with Gemini")
        
        prompt = f"""Extract comprehensive medical data for "{syndrome_name}" from pages {page_start}-{page_end}.

**EXTRACTION PRIORITIES:**

1. **PATHOGENS** (CRITICAL - Don't Miss Any!):
   • Search sections: "Etiology", "Causative Organisms", "Microbiology", "Common Pathogens"
   • Extract EVERY organism mentioned - in text AND tables
   • Include scientific names (Genus species format)
   • Capture antimicrobial resistance: ESBL, MRSA, CRE, MDR, VRE
   • Note prevalence if stated: "most common", "20-30%", "rare"
   • If an organism is mentioned, include it even without prevalence

2. **DRUGS** (CRITICAL - Get Complete Dosing):
   • Look for treatment tables and dosing schedules
   • FIRST-LINE drugs: marked as "recommended", "empirical", "initial", "first choice"
   • ALTERNATIVE drugs: marked as "alternative", "second-line", "if resistant", "allergy"
   • For each drug extract:
     - Drug name (generic)
     - Dosage with units (2g, 500mg, 50mg/kg)
     - Route (IV, PO, IM, SC)
     - Duration (7-10 days, 4-6 weeks, "until afebrile 48h")
     - Frequency (every 12h, TID, once daily)
   • Extract from ALL treatment scenarios (mild/moderate/severe)

3. **DEFINITION & DIAGNOSIS**:
   • Clinical definition (comprehensive description)
   • Diagnostic criteria (how to diagnose)
   • ICD-10 codes if mentioned

4. **SPECIAL POPULATIONS** (if mentioned):
   • Pregnancy modifications (note trimester)
   • Pediatric dosing (note age group)
   • Renal/hepatic adjustments

5. **CONTRAINDICATIONS** (if mentioned):
   • Absolute or relative
   • Specific conditions

**EXTRACTION STRATEGY:**
- Scan ALL tables completely - don't stop at first few rows
- Look for "continued on next page" - check subsequent pages
- Extract organism names in italic text (these are scientific names)
- If a table shows multiple drug regimens, extract ALL of them
- For incomplete information, provide what's available (use null for missing data)

Be thorough and extract ALL visible information from the PDF pages!
"""

        try:
            # Send PDF directly to Gemini with structured output
            # Using correct Google GenAI SDK API
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    types.Part.from_bytes(
                        data=self.pdf_bytes,
                        mime_type='application/pdf',
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=SyndromeData
                )
            )
            
            # Get the parsed Pydantic object directly
            syndrome_data: SyndromeData = response.parsed
            
            if syndrome_data is None:
                logger.warning("Gemini returned None for parsed data, trying text fallback")
                # Fallback to text parsing
                response_text = response.text
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                data = json.loads(response_text)
            else:
                # Convert Pydantic model to dict
                data = syndrome_data.model_dump()
            
            # Add metadata
            data['source_page'] = page_start
            data['source_document'] = self.pdf_path.name
            data['extracted_with'] = 'gemini-native-pdf'
            data['extraction_date'] = datetime.now().isoformat()
            
            logger.info(f"✓ Extracted: {len(data.get('common_pathogens', []))} pathogens, "
                      f"{len(data.get('first_line_drugs', []))} drugs")
            
            return data

        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            # Try to extract what we can
            try:
                response_text = response.text
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                data = json.loads(response_text)
                data['source_page'] = page_start
                data['source_document'] = self.pdf_path.name
                data['extracted_with'] = 'gemini-native-pdf'
                data['extraction_date'] = datetime.now().isoformat()
                logger.warning(f"Using raw data due to validation error")
                return data
            except Exception:
                return None
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def reprocess_all_failed(
        self,
        syndromes: List[Dict],
        failed_indices: List[int]
    ) -> Dict[int, Dict]:
        """
        Reprocess all failed syndromes with Gemini.
        
        Args:
            syndromes: Original syndrome list
            failed_indices: Indices to reprocess
            
        Returns:
            Dict of {index: new_syndrome_data}
        """
        reprocessed = {}
        
        from tqdm import tqdm
        
        for idx in tqdm(failed_indices, desc="Reprocessing with Gemini"):
            syndrome = syndromes[idx]
            syndrome_name = syndrome.get('syndrome_name', 'Unknown')
            page_start = syndrome.get('source_page', 0)
            page_end = page_start + 10  # Process 10-page window for context
            
            new_data = self.extract_with_gemini(syndrome_name, page_start, page_end)
            
            if new_data:
                # Compare with original
                old_pathogen_count = len(syndrome.get('common_pathogens', []))
                new_pathogen_count = len(new_data.get('common_pathogens', []))
                old_drug_count = len(syndrome.get('first_line_drugs', []))
                new_drug_count = len(new_data.get('first_line_drugs', []))
                
                logger.info(f"  Improvement: +{new_pathogen_count - old_pathogen_count} pathogens, "
                          f"+{new_drug_count - old_drug_count} drugs")
                
                reprocessed[idx] = new_data
            else:
                logger.warning(f"  Failed to reprocess {syndrome_name}")
        
        return reprocessed
    
    def merge_and_save(
        self,
        original_syndromes: List[Dict],
        reprocessed_data: Dict[int, Dict],
        output_file: str
    ):
        """
        Merge original and reprocessed data, then save.
        
        Args:
            original_syndromes: Original syndrome list
            reprocessed_data: Reprocessed syndromes
            output_file: Output file path
        """
        merged = []
        
        for i, syndrome in enumerate(original_syndromes):
            if i in reprocessed_data:
                merged.append(reprocessed_data[i])
            else:
                merged.append(syndrome)
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved merged results to {output_file}")
        
        # Generate report
        self._generate_report(original_syndromes, merged, output_file)
    
    def _generate_report(self, original: List[Dict], merged: List[Dict], output_file: str):
        """Generate comparison report."""
        
        def calc_stats(syndromes):
            missing_pathogens = sum(1 for s in syndromes if len(s.get('common_pathogens', [])) == 0)
            missing_drugs = sum(1 for s in syndromes if len(s.get('first_line_drugs', [])) == 0)
            total_pathogens = sum(len(s.get('common_pathogens', [])) for s in syndromes)
            total_drugs = sum(len(s.get('first_line_drugs', [])) for s in syndromes)
            return missing_pathogens, missing_drugs, total_pathogens, total_drugs
        
        orig_stats = calc_stats(original)
        merged_stats = calc_stats(merged)
        
        report = f"""
{'='*80}
GEMINI NATIVE PDF REPROCESSING REPORT
{'='*80}

BEFORE (Text-only extraction):
  Missing pathogens: {orig_stats[0]}
  Missing drugs: {orig_stats[1]}
  Total pathogens: {orig_stats[2]}
  Total drugs: {orig_stats[3]}

AFTER (Gemini native PDF):
  Missing pathogens: {merged_stats[0]}
  Missing drugs: {merged_stats[1]}
  Total pathogens: {merged_stats[2]}
  Total drugs: {merged_stats[3]}

IMPROVEMENTS:
  Syndromes with pathogens fixed: {orig_stats[0] - merged_stats[0]}
  Syndromes with drugs fixed: {orig_stats[1] - merged_stats[1]}
  Additional pathogens: +{merged_stats[2] - orig_stats[2]}
  Additional drugs: +{merged_stats[3] - orig_stats[3]}

{'='*80}
"""
        
        print(report)
        
        report_file = Path(output_file).parent / "gemini_reprocessing_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reprocess failed syndromes using Gemini native PDF processing'
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
        default='data/structured/all_syndromes_gemini.json',
        help='Output JSON file for merged results'
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        required=True,
        help='Path to ICMR PDF'
    )
    
    parser.add_argument(
        '--gemini-key',
        type=str,
        help='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    gemini_api_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ ERROR: No Gemini API key provided")
        print("Set GEMINI_API_KEY environment variable or use --gemini-key")
        return 1
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 80)
    print("GEMINI NATIVE PDF REPROCESSING")
    print("=" * 80)
    print()
    print("This uses Gemini's native PDF understanding to extract missing data.")
    print("Gemini sees the actual PDF layout, tables, and formatting.")
    print()
    
    try:
        # Initialize reprocessor
        reprocessor = GeminiPDFReprocessor(gemini_api_key, args.pdf)
        
        # Load existing results
        print("Step 1: Loading existing results...")
        syndromes = reprocessor.load_existing_results(args.input)
        print(f"✓ Loaded {len(syndromes)} syndromes\n")
        
        # Identify failures
        print("Step 2: Identifying failed syndromes...")
        failed_indices = reprocessor.identify_failed_syndromes(syndromes)
        print(f"✓ Found {len(failed_indices)} syndromes to reprocess\n")
        
        if not failed_indices:
            print("✅ All syndromes look good!")
            return 0
        
        # Reprocess with Gemini
        print("Step 3: Reprocessing with Gemini (native PDF)...")
        print("This may take a few minutes...")
        reprocessed = reprocessor.reprocess_all_failed(syndromes, failed_indices)
        print(f"\n✓ Reprocessed {len(reprocessed)} syndromes\n")
        
        # Merge and save
        print("Step 4: Merging and saving...")
        reprocessor.merge_and_save(syndromes, reprocessed, args.output)
        print(f"✓ Saved to {args.output}\n")
        
        print("=" * 80)
        print("✅ REPROCESSING COMPLETE")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
