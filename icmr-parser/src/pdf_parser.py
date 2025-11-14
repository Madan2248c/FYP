"""
PDF Parser for extracting text and tables from ICMR guideline PDFs.
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ICMRPDFParser:
    """Parser for ICMR antimicrobial guideline PDFs."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF parser.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        try:
            self.doc = fitz.open(str(self.pdf_path))
            self.total_pages = len(self.doc)
            self.extracted_content = []
            logger.info(f"Opened PDF: {self.pdf_path.name} with {self.total_pages} pages")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    def extract_text_and_tables(self, page_number: int) -> Dict[str, Any]:
        """
        Extract text and tables from a specific page.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Dictionary containing page content
        """
        try:
            page = self.doc[page_number]
            
            # Extract plain text
            text = page.get_text("text")
            
            # Extract tables
            tables = []
            try:
                tabs = page.find_tables()
                for i, table in enumerate(tabs):
                    df = table.to_pandas()
                    if not df.empty:
                        table_dict = {
                            'headers': df.columns.tolist(),
                            'rows': df.values.tolist(),
                            'bbox': table.bbox,
                            'table_number': i
                        }
                        tables.append(table_dict)
            except Exception as e:
                logger.warning(f"Table extraction failed on page {page_number}: {e}")
            
            # Get page dimensions
            rect = page.rect
            dimensions = {'width': rect.width, 'height': rect.height}
            
            return {
                'page_number': page_number,
                'text': text,
                'tables': tables,
                'dimensions': dimensions
            }
        except Exception as e:
            logger.error(f"Failed to extract content from page {page_number}: {e}")
            return {
                'page_number': page_number,
                'text': '',
                'tables': [],
                'dimensions': {},
                'error': str(e)
            }
    
    def identify_syndrome_sections(self) -> List[Dict[str, Any]]:
        """
        Identify sections in the PDF that correspond to syndromes.
        
        Returns:
            List of syndrome sections with their page ranges
        """
        logger.info("Identifying syndrome sections...")
        
        # Keywords that indicate syndrome sections
        syndrome_keywords = [
            'SYNDROME', 'INFECTION', 'FEVER', 'SEPSIS', 'PNEUMONIA',
            'URINARY TRACT', 'SKIN AND SOFT TISSUE', 'BONE AND JOINT',
            'CNS', 'CENTRAL NERVOUS SYSTEM', 'INTRA-ABDOMINAL', 'PELVIC',
            'BLOODSTREAM', 'MENINGITIS', 'ENCEPHALITIS', 'ENDOCARDITIS',
            'PERITONITIS', 'CHOLECYSTITIS', 'APPENDICITIS', 'CELLULITIS',
            'ABSCESS', 'OSTEOMYELITIS', 'ARTHRITIS', 'PYELONEPHRITIS',
            'CYSTITIS', 'PROSTATITIS', 'GASTROENTERITIS', 'DIARRHEA'
        ]
        
        sections = []
        current_section = None
        
        for page_num in range(self.total_pages):
            page_data = self.extract_text_and_tables(page_num)
            text = page_data['text']
            
            # Check first 500 characters for section headers
            first_chars = text[:500].upper()
            
            # Look for syndrome keywords in headers
            found_syndrome = False
            for keyword in syndrome_keywords:
                if keyword in first_chars:
                    # Check if it's likely a header (all caps, short line, or numbered)
                    lines = text.split('\n')[:5]
                    for line in lines:
                        if keyword in line.upper() and len(line.strip()) < 100:
                            # Found a new syndrome section
                            if current_section:
                                # Save previous section
                                sections.append(current_section)
                            
                            # Extract section number if present
                            section_match = re.search(r'(\d+\.?\d*)\s*', line)
                            section_number = section_match.group(1) if section_match else None
                            
                            current_section = {
                                'syndrome_name': line.strip(),
                                'page_start': page_num,
                                'page_end': page_num,
                                'full_text': text,
                                'tables': page_data['tables'],
                                'section_number': section_number
                            }
                            found_syndrome = True
                            break
                    if found_syndrome:
                        break
            
            # If no new section found, add to current section
            if not found_syndrome and current_section:
                current_section['page_end'] = page_num
                current_section['full_text'] += '\n\n' + text
                current_section['tables'].extend(page_data['tables'])
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        logger.info(f"Found {len(sections)} syndrome sections")
        return sections
    
    def extract_tables_as_dataframe(self, table_dict: Dict) -> pd.DataFrame:
        """
        Convert table dictionary to pandas DataFrame.
        
        Args:
            table_dict: Table dictionary from extract_text_and_tables
            
        Returns:
            Cleaned pandas DataFrame
        """
        try:
            df = pd.DataFrame(table_dict['rows'], columns=table_dict['headers'])
            
            # Clean column names
            df.columns = [str(col).strip().replace('\n', ' ') for col in df.columns]
            
            # Handle missing values
            df = df.fillna('')
            
            return df
        except Exception as e:
            logger.error(f"Failed to convert table to DataFrame: {e}")
            return pd.DataFrame()
    
    def save_extracted_content(self, output_dir: str) -> None:
        """
        Save extracted content to files.
        
        Args:
            output_dir: Directory to save extracted content
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save page-by-page content
        all_pages = []
        for page_num in range(self.total_pages):
            page_data = self.extract_text_and_tables(page_num)
            all_pages.append(page_data)
            
            # Save individual page text
            page_file = output_path / f"page_{page_num:03d}_text.json"
            with open(page_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'page_number': page_num,
                    'text': page_data['text']
                }, f, indent=2, ensure_ascii=False)
            
            # Save tables as CSV
            for i, table in enumerate(page_data['tables']):
                df = self.extract_tables_as_dataframe(table)
                if not df.empty:
                    csv_file = output_path / f"page_{page_num:03d}_table_{i}.csv"
                    df.to_csv(csv_file, index=False)
        
        # Save metadata
        syndrome_sections = self.identify_syndrome_sections()
        metadata = {
            'total_pages': self.total_pages,
            'source_file': str(self.pdf_path),
            'syndromes_found': len(syndrome_sections),
            'syndrome_list': [
                {
                    'name': s['syndrome_name'],
                    'page_start': s['page_start'],
                    'page_end': s['page_end'],
                    'section_number': s['section_number']
                }
                for s in syndrome_sections
            ]
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted content to {output_path}")
    
    def close(self) -> None:
        """Close the PDF document."""
        if hasattr(self, 'doc'):
            self.doc.close()
            logger.info("PDF document closed")

