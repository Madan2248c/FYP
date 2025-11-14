"""
Utility functions for ICMR parser.
"""

import logging
import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import fitz


def setup_logging() -> None:
    """Configure logging to write to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"icmr_extraction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary of configuration values
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()
    except Exception as e:
        logging.error(f"Failed to load config: {e}. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration."""
    return {
        'pdf_parsing': {
            'enable_table_detection': True,
            'min_table_rows': 2,
            'ocr_fallback': False
        },
        'llm': {
            'default_model': 'llama-3.1-70b-versatile',
            'temperature': 0,
            'max_retries': 3,
            'timeout_seconds': 60
        },
        'output': {
            'save_intermediate': True,
            'create_backup': True
        }
    }


def validate_pdf_file(pdf_path: str) -> Tuple[bool, str]:
    """
    Validate PDF file exists and can be opened.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    pdf_file = Path(pdf_path)
    
    # Check existence
    if not pdf_file.exists():
        return False, f"PDF file not found: {pdf_path}"
    
    # Check extension
    if pdf_file.suffix.lower() != '.pdf':
        return False, f"File is not a PDF: {pdf_path}"
    
    # Try to open with PyMuPDF
    try:
        doc = fitz.open(str(pdf_file))
        if len(doc) == 0:
            doc.close()
            return False, "PDF file has no pages"
        doc.close()
        return True, "Valid PDF file"
    except Exception as e:
        return False, f"Cannot open PDF: {str(e)}"


def clean_medical_text(text: str) -> str:
    """
    Clean and normalize medical text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common OCR errors
    replacements = {
        '|': 'I',  # Vertical bar to I (in Roman numerals)
        '0': 'O',  # Zero to O in specific contexts (would need more context)
    }
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize drug name to uppercase standard format.
    
    Args:
        drug_name: Raw drug name
        
    Returns:
        Normalized drug name
    """
    if not drug_name:
        return ""
    
    # Convert to uppercase
    normalized = drug_name.upper().strip()
    
    # Remove parenthetical brand names
    normalized = re.sub(r'\([^)]*\)', '', normalized).strip()
    
    return normalized


def merge_split_tables(tables_list: List[Dict]) -> List[Dict]:
    """
    Detect and merge tables split across pages.
    
    Args:
        tables_list: List of table dictionaries
        
    Returns:
        Merged table list
    """
    if len(tables_list) < 2:
        return tables_list
    
    merged = []
    i = 0
    
    while i < len(tables_list):
        current_table = tables_list[i]
        
        # Check if next table has matching headers
        if i + 1 < len(tables_list):
            next_table = tables_list[i + 1]
            
            current_headers = current_table.get('headers', [])
            next_headers = next_table.get('headers', [])
            
            # If headers match, merge
            if current_headers == next_headers:
                merged_table = {
                    'headers': current_headers,
                    'rows': current_table.get('rows', []) + next_table.get('rows', []),
                    'bbox': current_table.get('bbox'),
                    'table_number': current_table.get('table_number'),
                    'merged': True
                }
                merged.append(merged_table)
                i += 2  # Skip both tables
                continue
        
        merged.append(current_table)
        i += 1
    
    return merged


def load_groq_api_keys() -> List[str]:
    """
    Load all Groq API keys from environment variables.
    
    Returns:
        List of API keys
    """
    import os
    
    api_keys = []
    i = 1
    
    # Try to load keys numbered 1-20 (to be flexible)
    while i <= 20:
        key = os.getenv(f'GROQ_API_KEY_{i}')
        if key:
            api_keys.append(key)
            i += 1
        else:
            break
    
    # Also check for single key (backward compatibility)
    if not api_keys:
        single_key = os.getenv('GROQ_API_KEY')
        if single_key:
            api_keys.append(single_key)
    
    return api_keys


def estimate_extraction_cost(num_pages: int, model_name: str = "llama-3.3-70b-versatile") -> Dict[str, float]:
    """
    Estimate API cost for extraction.
    
    Args:
        num_pages: Number of pages to process
        model_name: Model name
        
    Returns:
        Dictionary with cost estimates
    """
    # Average tokens per page
    avg_input_tokens_per_page = 2000
    avg_output_tokens_per_page = 1000
    
    total_input_tokens = num_pages * avg_input_tokens_per_page
    total_output_tokens = num_pages * avg_output_tokens_per_page
    
    # Groq pricing (as of 2024 - verify current pricing)
    # Llama models on Groq are much cheaper than OpenAI
    if "llama" in model_name.lower():
        # Groq often has free tier or very low pricing
        input_cost_per_1m = 0.05  # $0.05 per 1M tokens (example)
        output_cost_per_1m = 0.08  # $0.08 per 1M tokens (example)
    else:
        input_cost_per_1m = 0.10
        output_cost_per_1m = 0.15
    
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost_usd': round(input_cost, 4),
        'output_cost_usd': round(output_cost, 4),
        'total_cost_usd': round(total_cost, 4)
    }


def create_validation_report(syndrome_data_list: List, output_file: str) -> None:
    """
    Create a validation report for extracted data.
    
    Args:
        syndrome_data_list: List of SyndromeData objects
        output_file: Path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ICMR EXTRACTION VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Overall statistics
    report_lines.append(f"Total syndromes extracted: {len(syndrome_data_list)}")
    
    total_drugs = sum(len(s.first_line_drugs) for s in syndrome_data_list)
    total_alternatives = sum(len(s.alternative_drugs) if s.alternative_drugs else 0 for s in syndrome_data_list)
    total_pathogens = sum(len(s.common_pathogens) for s in syndrome_data_list)
    
    report_lines.append(f"Total first-line drugs catalogued: {total_drugs}")
    report_lines.append(f"Total alternative drugs catalogued: {total_alternatives}")
    report_lines.append(f"Total pathogens identified: {total_pathogens}")
    
    if len(syndrome_data_list) > 0:
        avg_drugs = total_drugs / len(syndrome_data_list)
        report_lines.append(f"Average first-line drugs per syndrome: {avg_drugs:.2f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SYNDROME-BY-SYNDROME SUMMARY")
    report_lines.append("=" * 80 + "\n")
    
    # Individual syndrome validation
    for i, syndrome in enumerate(syndrome_data_list, 1):
        report_lines.append(f"{i}. {syndrome.syndrome_name}")
        report_lines.append(f"   Page: {syndrome.source_page}")
        report_lines.append(f"   Pathogens: {len(syndrome.common_pathogens)}")
        report_lines.append(f"   First-line drugs: {len(syndrome.first_line_drugs)}")
        report_lines.append(f"   Alternative drugs: {len(syndrome.alternative_drugs) if syndrome.alternative_drugs else 0}")
        
        # Validation checks
        issues = []
        
        if not syndrome.syndrome_name or len(syndrome.syndrome_name) < 3:
            issues.append("Missing or invalid syndrome name")
        
        if len(syndrome.first_line_drugs) == 0:
            issues.append("No first-line drugs found")
        
        if len(syndrome.common_pathogens) == 0:
            issues.append("No pathogens identified")
        
        # Check drug completeness
        for drug in syndrome.first_line_drugs:
            if not drug.dosage or not drug.route or not drug.duration:
                issues.append(f"Incomplete drug info: {drug.drug_name}")
                break
        
        if issues:
            report_lines.append(f"   ⚠️  ISSUES: {'; '.join(issues)}")
        else:
            report_lines.append("   ✓ Valid")
        
        report_lines.append("")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Validation report saved to {output_file}")

