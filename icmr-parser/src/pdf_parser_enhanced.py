"""
Enhanced PDF Parser for extracting text and tables from ICMR guideline PDFs.
Version 2.0 - Improved pathogen and treatment data extraction
"""

import fitz  # PyMuPDF
import pandas as pd
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ICMRPDFParserEnhanced:
    """
    Enhanced parser for ICMR antimicrobial guideline PDFs.
    
    Improvements over v1:
    - Better table detection (lower threshold, whitespace analysis)
    - Specialized pathogen and treatment extractors
    - Text pattern recognition for organisms and AMR patterns
    - Improved multi-page section grouping
    - Quality validation checks
    """
    
    # Pathogen-related keywords
    PATHOGEN_KEYWORDS = [
        'etiology', 'causative organisms', 'common pathogens', 'microbiology',
        'bacterial', 'viral', 'fungal', 'organism', 'pathogen', 'microorganism',
        'causative agent', 'etiological agent'
    ]
    
    # Treatment-related keywords
    TREATMENT_KEYWORDS = [
        'drug', 'antibiotic', 'antimicrobial', 'dosage', 'route', 'duration',
        'treatment', 'therapy', 'regimen', 'first-line', 'alternative',
        'empirical', 'definitive', 'medication'
    ]
    
    # Scientific name patterns
    ORGANISM_PATTERNS = [
        r'\b[A-Z][a-z]+\s+[a-z]+\b',  # Genus species
        r'\b[A-Z]\.\s*[a-z]+\b',       # G. species
    ]
    
    # AMR pattern indicators
    AMR_PATTERNS = [
        r'ESBL[-\s]producing', r'MRSA', r'VRE', r'CRE',
        r'carbapenem[-\s]resistant', r'penicillin[-\s]resistant',
        r'methicillin[-\s]resistant', r'vancomycin[-\s]resistant',
        r'multi[-\s]drug[-\s]resistant', r'MDR',
        r'extended[-\s]spectrum', r'beta[-\s]lactamase'
    ]
    
    def __init__(self, pdf_path: str):
        """Initialize the enhanced PDF parser."""
        self.pdf_path = Path(pdf_path)
        try:
            self.doc = fitz.open(str(self.pdf_path))
            self.total_pages = len(self.doc)
            self.extracted_content = []
            logger.info(f"Opened PDF: {self.pdf_path.name} with {self.total_pages} pages")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    def extract_text_and_tables_enhanced(self, page_number: int) -> Dict[str, Any]:
        """
        Enhanced extraction with better table detection.
        
        Args:
            page_number: Page number (0-indexed)
            
        Returns:
            Dictionary containing page content with enhanced table detection
        """
        try:
            page = self.doc[page_number]
            
            # Extract plain text
            text = page.get_text("text")
            
            # Extract text with formatting info (for detecting italic/bold)
            text_dict = page.get_text("dict")
            
            # Extract tables with standard detection
            tables = []
            try:
                # Lower threshold for better detection
                tabs = page.find_tables(strategy="lines_strict")
                for i, table in enumerate(tabs):
                    df = table.to_pandas()
                    if not df.empty and len(df) > 0:
                        table_dict = {
                            'headers': df.columns.tolist(),
                            'rows': df.values.tolist(),
                            'bbox': table.bbox,
                            'table_number': i,
                            'extraction_method': 'standard'
                        }
                        tables.append(table_dict)
            except Exception as e:
                logger.debug(f"Standard table extraction on page {page_number}: {e}")
            
            # Try whitespace-based table detection as fallback
            whitespace_tables = self._detect_whitespace_tables(text, page_number)
            tables.extend(whitespace_tables)
            
            # Extract italic text (likely organism names)
            italic_text = self._extract_italic_text(text_dict)
            
            # Get page dimensions
            rect = page.rect
            dimensions = {'width': rect.width, 'height': rect.height}
            
            return {
                'page_number': page_number,
                'text': text,
                'tables': tables,
                'italic_text': italic_text,
                'dimensions': dimensions,
                'text_dict': text_dict
            }
        except Exception as e:
            logger.error(f"Failed to extract content from page {page_number}: {e}")
            return {
                'page_number': page_number,
                'text': '',
                'tables': [],
                'italic_text': [],
                'dimensions': {},
                'error': str(e)
            }
    
    def _detect_whitespace_tables(self, text: str, page_number: int) -> List[Dict]:
        """
        Detect tables based on whitespace alignment (for tables without borders).
        
        Args:
            text: Page text
            page_number: Page number
            
        Returns:
            List of detected tables
        """
        tables = []
        lines = text.split('\n')
        
        # Look for aligned columns (multiple spaces between words)
        potential_table_lines = []
        for line in lines:
            if re.search(r'\w+\s{2,}\w+\s{2,}\w+', line):  # At least 3 columns
                potential_table_lines.append(line)
        
        if len(potential_table_lines) >= 3:  # At least header + 2 rows
            try:
                # Parse aligned columns
                rows = []
                for line in potential_table_lines:
                    cols = re.split(r'\s{2,}', line.strip())
                    if len(cols) >= 2:
                        rows.append(cols)
                
                if rows:
                    headers = rows[0]
                    data_rows = rows[1:]
                    
                    table_dict = {
                        'headers': headers,
                        'rows': data_rows,
                        'bbox': None,
                        'table_number': len(tables),
                        'extraction_method': 'whitespace'
                    }
                    tables.append(table_dict)
                    logger.debug(f"Detected whitespace table on page {page_number}")
            except Exception as e:
                logger.debug(f"Whitespace table detection failed on page {page_number}: {e}")
        
        return tables
    
    def _extract_italic_text(self, text_dict: Dict) -> List[str]:
        """
        Extract text formatted in italics (often used for organism names).
        
        Args:
            text_dict: Text dictionary from PyMuPDF
            
        Returns:
            List of italic text strings
        """
        italic_texts = []
        
        try:
            for block in text_dict.get('blocks', []):
                if 'lines' in block:
                    for line in block['lines']:
                        for span in line.get('spans', []):
                            # Check if font contains 'Italic' or similar
                            font = span.get('font', '').lower()
                            if 'italic' in font or 'oblique' in font:
                                text = span.get('text', '').strip()
                                if len(text) > 2:  # Skip very short text
                                    italic_texts.append(text)
        except Exception as e:
            logger.debug(f"Failed to extract italic text: {e}")
        
        return italic_texts
    
    def extract_pathogen_section(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pathogen-specific information from a page.
        
        Args:
            page_data: Page data dictionary
            
        Returns:
            Dictionary containing pathogen information
        """
        text = page_data.get('text', '')
        tables = page_data.get('tables', [])
        italic_text = page_data.get('italic_text', [])
        
        pathogen_info = {
            'organisms_found': [],
            'amr_patterns': [],
            'pathogen_tables': [],
            'text_mentions': []
        }
        
        # 1. Check for pathogen-related section headers
        text_lower = text.lower()
        for keyword in self.PATHOGEN_KEYWORDS:
            if keyword in text_lower:
                # Extract text around the keyword
                pattern = rf'(.{{0,100}}{re.escape(keyword)}.{{0,200}})'
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    pathogen_info['text_mentions'].append(match.group(0))
        
        # 2. Identify pathogen-related tables
        for table in tables:
            headers = [str(h).lower() for h in table.get('headers', [])]
            if any(keyword in ' '.join(headers) for keyword in 
                   ['organism', 'pathogen', 'bacteria', 'virus', 'fungus', 'microorganism']):
                pathogen_info['pathogen_tables'].append(table)
        
        # 3. Extract organism names (scientific names)
        organisms = set()
        
        # From italic text
        for italic in italic_text:
            for pattern in self.ORGANISM_PATTERNS:
                matches = re.findall(pattern, italic)
                organisms.update(matches)
        
        # From regular text
        for pattern in self.ORGANISM_PATTERNS:
            matches = re.findall(pattern, text)
            organisms.update(matches)
        
        pathogen_info['organisms_found'] = list(organisms)
        
        # 4. Extract AMR patterns
        amr_patterns = set()
        for pattern in self.AMR_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amr_patterns.update(matches)
        
        pathogen_info['amr_patterns'] = list(amr_patterns)
        
        return pathogen_info
    
    def extract_treatment_tables(self, page_data: Dict[str, Any]) -> List[Dict]:
        """
        Extract treatment-related tables from a page.
        
        Args:
            page_data: Page data dictionary
            
        Returns:
            List of treatment tables
        """
        tables = page_data.get('tables', [])
        treatment_tables = []
        
        for table in tables:
            headers = [str(h).lower() for h in table.get('headers', [])]
            header_text = ' '.join(headers)
            
            # Check if table contains treatment-related columns
            treatment_indicators = 0
            for keyword in self.TREATMENT_KEYWORDS:
                if keyword in header_text:
                    treatment_indicators += 1
            
            # If at least 2 treatment keywords found, it's likely a treatment table
            if treatment_indicators >= 2:
                table['table_type'] = 'treatment'
                treatment_tables.append(table)
                logger.debug(f"Identified treatment table with {treatment_indicators} indicators")
        
        return treatment_tables
    
    def identify_syndrome_sections_enhanced(self) -> List[Dict[str, Any]]:
        """
        Enhanced syndrome section identification with better page grouping.
        
        Returns:
            List of syndrome sections with enhanced information
        """
        logger.info("Identifying syndrome sections (enhanced)...")
        
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
            page_data = self.extract_text_and_tables_enhanced(page_num)
            text = page_data['text']
            
            # Check for continuation indicators
            is_continuation = self._is_continuation_page(text)
            
            # Check first 500 characters for section headers
            first_chars = text[:500].upper()
            
            # Look for syndrome keywords in headers
            found_syndrome = False
            if not is_continuation:
                for keyword in syndrome_keywords:
                    if keyword in first_chars:
                        lines = text.split('\n')[:5]
                        for line in lines:
                            if keyword in line.upper() and len(line.strip()) < 100:
                                # Check if next page has tables (look ahead)
                                has_continuation = self._check_continuation(page_num)
                                
                                if current_section:
                                    # Validate and save previous section
                                    self._validate_section(current_section)
                                    sections.append(current_section)
                                
                                section_match = re.search(r'(\d+\.?\d*)\s*', line)
                                section_number = section_match.group(1) if section_match else None
                                
                                # Extract pathogen and treatment info
                                pathogen_info = self.extract_pathogen_section(page_data)
                                treatment_tables = self.extract_treatment_tables(page_data)
                                
                                current_section = {
                                    'syndrome_name': line.strip(),
                                    'page_start': page_num,
                                    'page_end': page_num,
                                    'full_text': text,
                                    'tables': page_data['tables'],
                                    'pathogen_info': pathogen_info,
                                    'treatment_tables': treatment_tables,
                                    'section_number': section_number,
                                    'has_continuation': has_continuation,
                                    'quality_score': 0  # Will be calculated later
                                }
                                found_syndrome = True
                                break
                        if found_syndrome:
                            break
            
            # Add to current section if no new section found
            if not found_syndrome and current_section:
                current_section['page_end'] = page_num
                current_section['full_text'] += '\n\n' + text
                current_section['tables'].extend(page_data['tables'])
                
                # Accumulate pathogen info
                page_pathogen_info = self.extract_pathogen_section(page_data)
                current_section['pathogen_info']['organisms_found'].extend(
                    page_pathogen_info['organisms_found']
                )
                current_section['pathogen_info']['amr_patterns'].extend(
                    page_pathogen_info['amr_patterns']
                )
                current_section['pathogen_info']['pathogen_tables'].extend(
                    page_pathogen_info['pathogen_tables']
                )
                
                # Accumulate treatment tables
                page_treatment_tables = self.extract_treatment_tables(page_data)
                current_section['treatment_tables'].extend(page_treatment_tables)
        
        # Add and validate the last section
        if current_section:
            self._validate_section(current_section)
            sections.append(current_section)
        
        logger.info(f"Found {len(sections)} syndrome sections (enhanced)")
        return sections
    
    def _is_continuation_page(self, text: str) -> bool:
        """Check if page is a continuation of previous section."""
        continuation_indicators = [
            r'\(cont\.?\)', r'\(continued\)', r'\.\.\.continued',
            r'continued from', r'table \d+ \(cont'
        ]
        
        first_lines = '\n'.join(text.split('\n')[:3]).lower()
        for pattern in continuation_indicators:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True
        return False
    
    def _check_continuation(self, page_num: int) -> bool:
        """Check if next page is a continuation."""
        if page_num + 1 < self.total_pages:
            try:
                next_page = self.doc[page_num + 1]
                next_text = next_page.get_text("text")
                return self._is_continuation_page(next_text)
            except Exception:
                return False
        return False
    
    def _validate_section(self, section: Dict) -> None:
        """
        Validate section quality and assign quality score.
        
        Args:
            section: Section dictionary to validate
        """
        quality_score = 0
        warnings = []
        
        # Check text length
        text_length = len(section.get('full_text', ''))
        if text_length < 100:
            warnings.append("Very short text content")
        else:
            quality_score += 1
        
        # Check for tables
        table_count = len(section.get('tables', []))
        if table_count == 0:
            warnings.append("No tables found")
        else:
            quality_score += 2
        
        # Check for pathogen information
        pathogen_info = section.get('pathogen_info', {})
        if pathogen_info.get('organisms_found'):
            quality_score += 2
        else:
            warnings.append("No organisms identified")
        
        if pathogen_info.get('pathogen_tables'):
            quality_score += 1
        
        # Check for treatment information
        treatment_tables = section.get('treatment_tables', [])
        if treatment_tables:
            quality_score += 2
        else:
            warnings.append("No treatment tables found")
        
        # Check for AMR patterns
        if pathogen_info.get('amr_patterns'):
            quality_score += 1
        
        section['quality_score'] = quality_score
        section['quality_warnings'] = warnings
        
        if warnings:
            logger.warning(f"Section '{section['syndrome_name']}' quality issues: {', '.join(warnings)}")
    
    def merge_continued_tables(self, sections: List[Dict]) -> List[Dict]:
        """
        Merge tables that continue across pages.
        
        Args:
            sections: List of syndrome sections
            
        Returns:
            Sections with merged tables
        """
        for section in sections:
            tables = section.get('tables', [])
            merged_tables = []
            skip_next = False
            
            for i, table in enumerate(tables):
                if skip_next:
                    skip_next = False
                    continue
                
                # Check if next table has matching headers
                if i + 1 < len(tables):
                    next_table = tables[i + 1]
                    
                    if self._tables_match(table, next_table):
                        # Merge tables
                        merged = self._merge_two_tables(table, next_table)
                        merged_tables.append(merged)
                        skip_next = True
                        logger.debug(f"Merged tables {i} and {i+1} in section {section['syndrome_name']}")
                        continue
                
                merged_tables.append(table)
            
            section['tables'] = merged_tables
        
        return sections
    
    def _tables_match(self, table1: Dict, table2: Dict) -> bool:
        """Check if two tables have matching headers (for merging)."""
        headers1 = [str(h).lower().strip() for h in table1.get('headers', [])]
        headers2 = [str(h).lower().strip() for h in table2.get('headers', [])]
        
        if not headers1 or not headers2:
            return False
        
        # Check if headers match (allow for slight variations)
        if len(headers1) != len(headers2):
            return False
        
        matches = sum(1 for h1, h2 in zip(headers1, headers2) if h1 == h2)
        return matches >= len(headers1) * 0.8  # 80% match threshold
    
    def _merge_two_tables(self, table1: Dict, table2: Dict) -> Dict:
        """Merge two tables with matching headers."""
        merged = table1.copy()
        merged['rows'] = table1['rows'] + table2['rows']
        merged['merged'] = True
        return merged
    
    def save_extracted_content_v2(self, output_dir: str) -> None:
        """
        Save enhanced extracted content with comparison to v1.
        
        Args:
            output_dir: Directory to save content (data/extracted_v2)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Extracting content with enhanced parser...")
        
        # Extract all pages
        all_pages = []
        for page_num in range(self.total_pages):
            page_data = self.extract_text_and_tables_enhanced(page_num)
            all_pages.append(page_data)
            
            # Save page text
            page_file = output_path / f"page_{page_num:03d}_enhanced.json"
            with open(page_file, 'w', encoding='utf-8') as f:
                # Remove text_dict to reduce file size
                save_data = {k: v for k, v in page_data.items() if k != 'text_dict'}
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # Get enhanced syndrome sections
        syndrome_sections = self.identify_syndrome_sections_enhanced()
        
        # Merge continued tables
        syndrome_sections = self.merge_continued_tables(syndrome_sections)
        
        # Save enhanced metadata
        metadata = {
            'total_pages': self.total_pages,
            'source_file': str(self.pdf_path),
            'syndromes_found': len(syndrome_sections),
            'extraction_version': '2.0',
            'enhancements': [
                'Improved table detection',
                'Pathogen section extraction',
                'Treatment table identification',
                'Text pattern recognition',
                'Quality validation',
                'Table merging'
            ],
            'syndrome_list': []
        }
        
        # Build syndrome list with quality metrics
        for s in syndrome_sections:
            syndrome_info = {
                'name': s['syndrome_name'],
                'page_start': s['page_start'],
                'page_end': s['page_end'],
                'section_number': s['section_number'],
                'quality_score': s['quality_score'],
                'quality_warnings': s['quality_warnings'],
                'table_count': len(s['tables']),
                'organisms_found': len(s['pathogen_info']['organisms_found']),
                'amr_patterns': len(s['pathogen_info']['amr_patterns']),
                'treatment_tables': len(s['treatment_tables']),
                'has_pathogen_table': len(s['pathogen_info']['pathogen_tables']) > 0
            }
            metadata['syndrome_list'].append(syndrome_info)
        
        # Save metadata
        metadata_file = output_path / "metadata_v2.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save detailed syndrome sections
        syndromes_file = output_path / "syndrome_sections_v2.json"
        with open(syndromes_file, 'w', encoding='utf-8') as f:
            json.dump(syndrome_sections, f, indent=2, ensure_ascii=False)
        
        # Generate comparison report
        self._generate_comparison_report(output_path, metadata)
        
        logger.info(f"Saved enhanced extracted content to {output_path}")
    
    def _generate_comparison_report(self, output_path: Path, metadata: Dict) -> None:
        """Generate a comparison report between v1 and v2 extraction."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ICMR PDF PARSER - V2 ENHANCEMENT REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("EXTRACTION SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total syndromes found: {metadata['syndromes_found']}")
        
        # Quality statistics
        avg_quality = sum(s['quality_score'] for s in metadata['syndrome_list']) / len(metadata['syndrome_list'])
        report_lines.append(f"Average quality score: {avg_quality:.2f} / 9")
        
        syndromes_with_organisms = sum(1 for s in metadata['syndrome_list'] if s['organisms_found'] > 0)
        report_lines.append(f"Syndromes with organisms identified: {syndromes_with_organisms} / {metadata['syndromes_found']}")
        
        syndromes_with_treatment = sum(1 for s in metadata['syndrome_list'] if s['treatment_tables'] > 0)
        report_lines.append(f"Syndromes with treatment tables: {syndromes_with_treatment} / {metadata['syndromes_found']}")
        
        syndromes_with_amr = sum(1 for s in metadata['syndrome_list'] if s['amr_patterns'] > 0)
        report_lines.append(f"Syndromes with AMR patterns: {syndromes_with_amr} / {metadata['syndromes_found']}")
        
        report_lines.append("")
        report_lines.append("DETAILED SYNDROME ANALYSIS")
        report_lines.append("-" * 80)
        
        for i, syndrome in enumerate(metadata['syndrome_list'], 1):
            report_lines.append(f"\n{i}. {syndrome['name']}")
            report_lines.append(f"   Pages: {syndrome['page_start']}-{syndrome['page_end']}")
            report_lines.append(f"   Quality Score: {syndrome['quality_score']}/9")
            report_lines.append(f"   Tables: {syndrome['table_count']}")
            report_lines.append(f"   Organisms: {syndrome['organisms_found']}")
            report_lines.append(f"   AMR Patterns: {syndrome['amr_patterns']}")
            report_lines.append(f"   Treatment Tables: {syndrome['treatment_tables']}")
            report_lines.append(f"   Has Pathogen Table: {'Yes' if syndrome['has_pathogen_table'] else 'No'}")
            
            if syndrome['quality_warnings']:
                report_lines.append(f"   ⚠️  Warnings: {', '.join(syndrome['quality_warnings'])}")
        
        # Save report
        report_file = output_path / "v2_enhancement_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated comparison report: {report_file}")
    
    def close(self) -> None:
        """Close the PDF document."""
        if hasattr(self, 'doc'):
            self.doc.close()
            logger.info("PDF document closed")

