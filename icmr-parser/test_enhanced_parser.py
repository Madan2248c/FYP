"""
Test script for enhanced PDF parser.
Compares v1 vs v2 extraction results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pdf_parser import ICMRPDFParser
from src.pdf_parser_enhanced import ICMRPDFParserEnhanced
from src.utils import setup_logging
import logging

logger = logging.getLogger(__name__)


def test_enhanced_parser(pdf_path: str):
    """Test the enhanced parser and compare with original."""
    
    print("=" * 80)
    print("TESTING ENHANCED PDF PARSER")
    print("=" * 80)
    print()
    
    # Test original parser (v1)
    print("Step 1: Running original parser (v1)...")
    print("-" * 80)
    
    try:
        parser_v1 = ICMRPDFParser(pdf_path)
        print(f"✓ Opened PDF: {parser_v1.total_pages} pages")
        
        sections_v1 = parser_v1.identify_syndrome_sections()
        print(f"✓ Found {len(sections_v1)} syndromes")
        
        # Save v1 results
        parser_v1.save_extracted_content('data/extracted')
        print(f"✓ Saved v1 extraction to data/extracted/")
        
        parser_v1.close()
    except Exception as e:
        print(f"✗ Error in v1 parser: {e}")
        return False
    
    print()
    
    # Test enhanced parser (v2)
    print("Step 2: Running enhanced parser (v2)...")
    print("-" * 80)
    
    try:
        parser_v2 = ICMRPDFParserEnhanced(pdf_path)
        print(f"✓ Opened PDF: {parser_v2.total_pages} pages")
        
        # Test enhanced features
        print("\nTesting enhanced features:")
        
        # Test single page extraction
        page_data = parser_v2.extract_text_and_tables_enhanced(20)
        print(f"  ✓ Enhanced extraction: {len(page_data['tables'])} tables, {len(page_data['italic_text'])} italic texts")
        
        # Test pathogen extraction
        pathogen_info = parser_v2.extract_pathogen_section(page_data)
        print(f"  ✓ Pathogen extraction: {len(pathogen_info['organisms_found'])} organisms, {len(pathogen_info['amr_patterns'])} AMR patterns")
        
        # Test treatment extraction
        treatment_tables = parser_v2.extract_treatment_tables(page_data)
        print(f"  ✓ Treatment extraction: {len(treatment_tables)} treatment tables")
        
        # Extract all sections
        print("\nExtracting all syndrome sections...")
        sections_v2 = parser_v2.identify_syndrome_sections_enhanced()
        print(f"✓ Found {len(sections_v2)} syndromes")
        
        # Save v2 results
        parser_v2.save_extracted_content_v2('data/extracted_v2')
        print(f"✓ Saved v2 extraction to data/extracted_v2/")
        
        parser_v2.close()
    except Exception as e:
        print(f"✗ Error in v2 parser: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Compare results
    print("Step 3: Comparing v1 vs v2...")
    print("-" * 80)
    
    print(f"Syndromes found:")
    print(f"  v1: {len(sections_v1)}")
    print(f"  v2: {len(sections_v2)}")
    print(f"  Difference: {len(sections_v2) - len(sections_v1)}")
    
    # Analyze v2 improvements
    total_organisms = sum(len(s['pathogen_info']['organisms_found']) for s in sections_v2)
    total_amr = sum(len(s['pathogen_info']['amr_patterns']) for s in sections_v2)
    total_treatment_tables = sum(len(s['treatment_tables']) for s in sections_v2)
    avg_quality = sum(s['quality_score'] for s in sections_v2) / len(sections_v2)
    
    print(f"\nv2 Enhancements:")
    print(f"  Total organisms identified: {total_organisms}")
    print(f"  Total AMR patterns found: {total_amr}")
    print(f"  Total treatment tables: {total_treatment_tables}")
    print(f"  Average quality score: {avg_quality:.2f} / 9")
    
    # Show sections with issues
    low_quality_sections = [s for s in sections_v2 if s['quality_score'] < 4]
    if low_quality_sections:
        print(f"\n⚠️  {len(low_quality_sections)} sections with quality concerns:")
        for s in low_quality_sections[:5]:  # Show first 5
            print(f"  - {s['syndrome_name']}: score {s['quality_score']}, warnings: {', '.join(s['quality_warnings'])}")
    
    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review data/extracted_v2/v2_enhancement_report.txt")
    print("2. Check data/extracted_v2/metadata_v2.json for details")
    print("3. Compare organism extraction: v2 should have more pathogen data")
    print()
    
    return True


def main():
    """Main test function."""
    setup_logging()
    
    pdf_path = "data/raw/NTG-Version-31st-July-final.pdf"
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please ensure the PDF is in data/raw/")
        return 1
    
    # Run test
    success = test_enhanced_parser(pdf_path)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

