"""
Main execution pipeline for ICMR antimicrobial guideline extraction.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pdf_parser import ICMRPDFParser
from src.llm_extractor import ICMRStructuredExtractor
from src.utils import (
    setup_logging,
    load_config,
    validate_pdf_file,
    estimate_extraction_cost,
    create_validation_report,
    load_groq_api_keys
)
import logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract structured data from ICMR antimicrobial guidelines'
    )
    
    parser.add_argument(
        '--pdf-path',
        type=str,
        required=True,
        help='Path to ICMR PDF file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/structured',
        help='Directory to save structured output (default: data/structured/)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama-3.3-70b-versatile',
        help='LLM model to use (default: llama-3.3-70b-versatile)'
    )
    
    parser.add_argument(
        '--skip-parsing',
        action='store_true',
        help='Skip PDF parsing if already done'
    )
    
    parser.add_argument(
        '--max-syndromes',
        type=int,
        default=None,
        help='Limit number of syndromes to process (for testing)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers for extraction (default: number of API keys, max 8)'
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    
    # Step 1: Setup
    print("=" * 80)
    print("ICMR ANTIMICROBIAL GUIDELINE EXTRACTOR")
    print("Using Groq API with Llama 70B")
    print("=" * 80)
    print()
    
    setup_logging()
    logger.info("Starting ICMR extraction pipeline")
    
    # Parse arguments
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    api_keys = load_groq_api_keys()
    
    if not api_keys:
        logger.error("No Groq API keys found in environment variables")
        print("\n❌ ERROR: No GROQ API keys found in .env file")
        print("Please create a .env file with your Groq API key(s):")
        print("GROQ_API_KEY_1=your_key_here")
        print("GROQ_API_KEY_2=your_key_here")
        print("...")
        return 1
    
    logger.info(f"Loaded {len(api_keys)} API key(s)")
    print(f"✓ Loaded {len(api_keys)} Groq API key(s) for rotation")
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate PDF file
    is_valid, message = validate_pdf_file(args.pdf_path)
    if not is_valid:
        logger.error(f"PDF validation failed: {message}")
        print(f"\n❌ ERROR: {message}")
        return 1
    
    logger.info(f"PDF validated: {args.pdf_path}")
    print(f"✓ PDF file validated: {args.pdf_path}\n")
    
    # Step 2: PDF Parsing
    syndrome_sections = []
    
    if not args.skip_parsing:
        print("Step 1: Parsing PDF...")
        print("-" * 80)
        
        try:
            parser = ICMRPDFParser(args.pdf_path)
            print(f"✓ Opened PDF: {parser.total_pages} pages")
            
            # Extract syndrome sections
            syndrome_sections = parser.identify_syndrome_sections()
            print(f"✓ Identified {len(syndrome_sections)} syndrome sections")
            
            # Save extracted content
            extracted_dir = 'data/extracted'
            parser.save_extracted_content(extracted_dir)
            print(f"✓ Saved extracted content to {extracted_dir}/")
            
            # Close parser
            parser.close()
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            print(f"\n❌ ERROR during PDF parsing: {e}")
            return 1
    else:
        print("Step 1: Skipping PDF parsing (--skip-parsing flag set)")
        print("Note: You'll need to load syndrome sections from saved data")
        print()
    
    # Limit syndromes if specified
    if args.max_syndromes:
        syndrome_sections = syndrome_sections[:args.max_syndromes]
        print(f"\n⚠️  Limited to {args.max_syndromes} syndromes for testing")
    
    print()
    
    # Step 3: Cost Estimation
    print("Step 2: Cost Estimation")
    print("-" * 80)
    
    num_sections = len(syndrome_sections)
    cost_info = estimate_extraction_cost(num_sections, args.model)
    
    print(f"Sections to process: {num_sections}")
    print(f"Estimated input tokens: {cost_info['total_input_tokens']:,}")
    print(f"Estimated output tokens: {cost_info['total_output_tokens']:,}")
    print(f"Estimated cost: ${cost_info['total_cost_usd']:.4f}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with extraction? (y/n): ").strip().lower()
    if response != 'y':
        print("Extraction cancelled by user.")
        return 0
    
    print()
    
    # Step 4: LLM Extraction
    print("Step 3: Extracting Structured Data")
    print("-" * 80)
    
    try:
        extractor = ICMRStructuredExtractor(
            api_keys=api_keys,
            model_name=args.model,
            temperature=config['llm']['temperature']
        )
        
        workers = args.workers if args.workers else min(len(api_keys), 8)
        
        print(f"Using model: {args.model}")
        print(f"Temperature: {config['llm']['temperature']}")
        print(f"API keys available: {len(api_keys)}")
        print(f"Parallel workers: {workers}")
        print(f"Starting PARALLEL extraction with automatic key rotation...\n")
        
        # Extract data with parallel processing
        extracted_data = extractor.batch_extract_syndromes(
            syndrome_sections_list=syndrome_sections,
            output_dir=args.output_dir,
            max_workers=workers
        )
        
        print(f"\n✓ Extracted {len(extracted_data)} syndromes successfully")
        
        # Save structured data
        extractor.save_structured_data(extracted_data, args.output_dir)
        print(f"✓ Saved structured data to {args.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        logger.warning("Extraction interrupted by user")
        print("Partial results may have been saved to intermediate files.")
        return 1
    
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        print(f"\n❌ ERROR during extraction: {e}")
        return 1
    
    print()
    
    # Step 5: Validation and Summary
    print("Step 4: Validation and Summary")
    print("-" * 80)
    
    if extracted_data:
        # Calculate statistics
        total_drugs = sum(len(s.first_line_drugs) for s in extracted_data)
        total_alternatives = sum(len(s.alternative_drugs) if s.alternative_drugs else 0 for s in extracted_data)
        total_pathogens = sum(len(s.common_pathogens) for s in extracted_data)
        avg_drugs = total_drugs / len(extracted_data) if extracted_data else 0
        
        print(f"Total syndromes extracted: {len(extracted_data)}")
        print(f"Total drugs catalogued: {total_drugs}")
        print(f"Total alternative drugs: {total_alternatives}")
        print(f"Total pathogens identified: {total_pathogens}")
        print(f"Average drugs per syndrome: {avg_drugs:.2f}")
        
        # Create validation report
        report_file = Path(args.output_dir) / "validation_report.txt"
        create_validation_report(extracted_data, str(report_file))
        print(f"\n✓ Validation report saved to {report_file}")
    else:
        print("⚠️  No data was extracted")
    
    print()
    
    # Step 6: Completion
    print("=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the validation report for any issues")
    print("2. Check data/structured/all_syndromes.json for complete data")
    print("3. Use the structured data to build your knowledge graph")
    print("4. Train your reasoning LLM with the extracted information")
    print()
    
    logger.info("Pipeline completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())

