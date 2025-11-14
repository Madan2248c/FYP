"""
Test script for ICMR extraction pipeline.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pdf_parser import ICMRPDFParser
from src.llm_extractor import ICMRStructuredExtractor
from src.data_models import SyndromeData, DrugRecommendation, Pathogen
from src.utils import setup_logging, validate_pdf_file, load_groq_api_keys

import logging

logger = logging.getLogger(__name__)


def test_pdf_parser(pdf_path: str):
    """Test PDF parser on a single page."""
    print("\n" + "=" * 80)
    print("TEST 1: PDF PARSER")
    print("=" * 80)
    
    try:
        parser = ICMRPDFParser(pdf_path)
        print(f"✓ Opened PDF: {parser.total_pages} pages")
        
        # Test extracting first page
        print("\nExtracting first page...")
        page_data = parser.extract_text_and_tables(0)
        
        print(f"✓ Page text length: {len(page_data['text'])} characters")
        print(f"✓ Tables found: {len(page_data['tables'])}")
        
        if page_data['text']:
            print("\nFirst 200 characters of text:")
            print("-" * 80)
            print(page_data['text'][:200])
            print("-" * 80)
        
        parser.close()
        print("\n✓ PDF Parser test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ PDF Parser test FAILED: {e}")
        return False


def test_pydantic_models():
    """Test Pydantic data models."""
    print("\n" + "=" * 80)
    print("TEST 2: PYDANTIC DATA MODELS")
    print("=" * 80)
    
    try:
        # Test creating a drug recommendation
        drug = DrugRecommendation(
            drug_name="AMOXICILLIN",
            dosage="500mg",
            route="PO",
            frequency="TID",
            duration="7 days",
            indication="first-line"
        )
        print(f"✓ Created DrugRecommendation: {drug.drug_name}")
        
        # Test creating a pathogen
        pathogen = Pathogen(
            organism_name="Streptococcus pneumoniae",
            prevalence="most common"
        )
        print(f"✓ Created Pathogen: {pathogen.organism_name}")
        
        # Test creating a syndrome
        syndrome = SyndromeData(
            syndrome_name="Community-Acquired Pneumonia",
            definition="Acute infection of lung parenchyma",
            common_pathogens=[pathogen],
            first_line_drugs=[drug],
            source_page=1,
            source_document="test.pdf"
        )
        print(f"✓ Created SyndromeData: {syndrome.syndrome_name}")
        
        # Test JSON serialization
        json_data = syndrome.model_dump()
        print(f"✓ Serialized to JSON: {len(str(json_data))} characters")
        
        print("\n✓ Pydantic Models test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Pydantic Models test FAILED: {e}")
        return False


def test_llm_extractor():
    """Test LLM extractor with sample text."""
    print("\n" + "=" * 80)
    print("TEST 3: LLM EXTRACTOR")
    print("=" * 80)
    
    # Load API keys
    load_dotenv()
    api_keys = load_groq_api_keys()
    
    if not api_keys:
        print("✗ No GROQ API keys found. Skipping LLM test.")
        return False
    
    print(f"✓ Found {len(api_keys)} API key(s)")
    
    sample_text = """
    COMMUNITY-ACQUIRED PNEUMONIA (CAP)
    
    Definition: Acute infection of the lung parenchyma acquired outside hospital settings.
    
    Common Pathogens:
    - Streptococcus pneumoniae (most common)
    - Haemophilus influenzae
    - Mycoplasma pneumoniae
    - Klebsiella pneumoniae (ESBL-producing strains increasing)
    
    First-Line Treatment:
    - Amoxicillin 500mg PO TID for 7 days (outpatient, non-severe)
    - Ceftriaxone 1g IV once daily for 7-10 days (hospitalized patients)
    - Add Azithromycin 500mg PO/IV once daily if atypical pathogens suspected
    
    Alternative Options:
    - Levofloxacin 750mg PO/IV once daily for 5 days (if beta-lactam allergy)
    - Moxifloxacin 400mg PO/IV once daily for 7 days
    
    Special Populations:
    - Pregnancy: Use Amoxicillin, avoid fluoroquinolones
    - Renal impairment: Adjust ceftriaxone dose
    
    Monitoring: Check renal function, inflammatory markers
    """
    
    try:
        extractor = ICMRStructuredExtractor(api_keys=api_keys)
        print(f"✓ Initialized LLM extractor with {len(api_keys)} key(s)")
        
        print("\nExtracting sample syndrome data...")
        print("(This will make an API call to Groq)")
        
        syndrome_data = extractor.extract_syndrome_data(
            syndrome_text=sample_text,
            tables_list=[],
            page_number=1,
            source_pdf="test.pdf"
        )
        
        if syndrome_data:
            print(f"\n✓ Extracted syndrome: {syndrome_data.syndrome_name}")
            print(f"✓ Pathogens found: {len(syndrome_data.common_pathogens)}")
            print(f"✓ First-line drugs: {len(syndrome_data.first_line_drugs)}")
            
            if syndrome_data.first_line_drugs:
                drug = syndrome_data.first_line_drugs[0]
                print(f"\nSample drug extracted:")
                print(f"  - Name: {drug.drug_name}")
                print(f"  - Dosage: {drug.dosage}")
                print(f"  - Route: {drug.route}")
                print(f"  - Frequency: {drug.frequency}")
                print(f"  - Duration: {drug.duration}")
            
            print("\n✓ LLM Extractor test PASSED")
            return True
        else:
            print("\n✗ LLM Extractor test FAILED: No data extracted")
            return False
        
    except Exception as e:
        print(f"\n✗ LLM Extractor test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    setup_logging()
    
    print("\n" + "=" * 80)
    print("ICMR EXTRACTION PIPELINE - TEST SUITE")
    print("=" * 80)
    
    # Check for PDF file
    pdf_path = "data/raw/NTG-Version-31st-July-final.pdf"
    
    is_valid, message = validate_pdf_file(pdf_path)
    if not is_valid:
        print(f"\n⚠️  Warning: {message}")
        print("Some tests will be skipped.")
        pdf_path = None
    else:
        print(f"\n✓ Found test PDF: {pdf_path}")
    
    # Run tests
    results = []
    
    # Test 1: PDF Parser
    if pdf_path:
        results.append(("PDF Parser", test_pdf_parser(pdf_path)))
    else:
        print("\n⊘ Skipping PDF Parser test (no PDF file)")
        results.append(("PDF Parser", None))
    
    # Test 2: Pydantic Models
    results.append(("Pydantic Models", test_pydantic_models()))
    
    # Test 3: LLM Extractor (optional, requires API key)
    ask_llm_test = input("\nRun LLM extractor test? This will use Groq API credits. (y/n): ")
    if ask_llm_test.strip().lower() == 'y':
        results.append(("LLM Extractor", test_llm_extractor()))
    else:
        print("\n⊘ Skipping LLM Extractor test")
        results.append(("LLM Extractor", None))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"{test_name:.<50} {status}")
    
    print("\n")
    
    # Exit code
    failed = sum(1 for _, result in results if result is False)
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

