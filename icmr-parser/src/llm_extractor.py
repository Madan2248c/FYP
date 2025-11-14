"""
LLM-based structured data extractor using Groq API with Llama 70B.
Uses LangChain's structured output for reliable extraction.
Enhanced with two-pass extraction and quality validation.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from langchain_groq import ChatGroq
from pydantic import ValidationError
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

from .data_models import SyndromeData

logger = logging.getLogger(__name__)


class ICMRStructuredExtractor:
    """Extractor for structured medical data using Groq's Llama 70B with automatic key rotation.
    
    Uses LangChain's ChatGroq with structured output for reliable JSON extraction.
    """
    
    def __init__(self, api_keys: List[str], model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3):
        """
        Initialize the LLM extractor with multiple API keys for automatic rotation.
        
        Args:
            api_keys: List of Groq API keys (will rotate on errors)
            model_name: Model name (default: llama-3.3-70b-versatile)
            temperature: Temperature for generation (default: 0.3)
        """
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model_name = model_name
        self.temperature = temperature
        self.failed_keys = set()  # Track keys that have failed
        
        # Thread safety for key rotation
        self.key_lock = threading.Lock()
        
        # Initialize ChatGroq with first key
        self._initialize_client()
        
        logger.info(f"Initialized LangChain ChatGroq with {len(api_keys)} API keys")
        logger.info(f"Model: {model_name}, Temperature: {temperature}")
        logger.info(f"Starting with API key #{self.current_key_index + 1}")
    
    def _initialize_client(self):
        """Initialize or reinitialize the ChatGroq client."""
        self.client = ChatGroq(
            groq_api_key=self.api_keys[self.current_key_index],
            model_name=self.model_name,
            temperature=self.temperature
        )
    
    def _rotate_api_key(self) -> bool:
        """
        Rotate to the next available API key (thread-safe).
        
        Returns:
            True if rotation successful, False if all keys exhausted
        """
        with self.key_lock:
            # Try each key once
            attempts = 0
            while attempts < len(self.api_keys):
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                
                # Skip if this key has permanently failed
                if self.current_key_index in self.failed_keys:
                    attempts += 1
                    continue
                
                # Try to initialize with new key
                try:
                    self._initialize_client()
                    logger.info(f"Rotated to API key #{self.current_key_index + 1}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize with key #{self.current_key_index + 1}: {e}")
                    self.failed_keys.add(self.current_key_index)
                    attempts += 1
            
            logger.error("All API keys have been exhausted or failed")
            return False
    
    def _call_structured_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Call LLM with JSON mode and automatic key rotation on failure.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum retry attempts per key
            
        Returns:
            Structured output as dictionary or None if all attempts fail
        """
        keys_tried = 0
        
        while keys_tried < len(self.api_keys):
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} with key #{self.current_key_index + 1}")
                    
                    # Use JSON mode instead of strict tool calling
                    # This is more flexible with Optional fields
                    response = self.client.invoke(
                        prompt,
                        response_format={"type": "json_object"}
                    )
                    
                    # Extract JSON from response
                    import json
                    import re
                    
                    content = response.content
                    
                    # Try to extract JSON from markdown code blocks
                    if "```json" in content:
                        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(1)
                    elif "```" in content:
                        json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(1)
                    
                    # Parse JSON
                    result_dict = json.loads(content)
                    
                    # Handle required fields - convert null to meaningful defaults
                    if not result_dict.get('syndrome_name'):
                        result_dict['syndrome_name'] = "Unnamed Syndrome"
                        logger.warning("syndrome_name was null or empty, using default")
                    
                    if not result_dict.get('definition'):
                        result_dict['definition'] = "Definition not provided in source document"
                        logger.warning("definition was null or empty, using default")
                    
                    # Ensure common_pathogens and first_line_drugs are lists
                    if not result_dict.get('common_pathogens'):
                        result_dict['common_pathogens'] = []
                    if not result_dict.get('first_line_drugs'):
                        result_dict['first_line_drugs'] = []
                    
                    # Convert "Not specified" strings to None for optional array fields
                    for field in ['icd10_codes', 'alternative_drugs', 'special_populations', 
                                  'contraindications', 'monitoring_requirements']:
                        if field in result_dict:
                            if result_dict[field] == "Not specified":
                                result_dict[field] = None
                            elif result_dict[field] == []:
                                result_dict[field] = None  # Convert empty arrays to None for optional fields
                    
                    # Convert "Not specified" or null to None for optional string fields
                    for field in ['diagnostic_criteria', 'de_escalation_guidance']:
                        if field in result_dict and (result_dict[field] == "Not specified" or not result_dict[field]):
                            result_dict[field] = None
                    
                    return result_dict
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Log full error details for debugging
                    logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
                    
                    # Check if it's a rate limit error
                    if "rate_limit" in error_msg or "429" in error_msg:
                        logger.warning(f"Rate limit hit on key #{self.current_key_index + 1}")
                        
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            # Exhausted retries on this key, rotate
                            logger.warning(f"Exhausted retries on key #{self.current_key_index + 1}, rotating...")
                            if not self._rotate_api_key():
                                return None
                            keys_tried += 1
                            break  # Break retry loop to try new key
                    
                    # Check if it's an authentication error
                    elif "authentication" in error_msg or "invalid" in error_msg or "401" in error_msg or "400" in error_msg:
                        logger.error(f"Authentication/Request failed for key #{self.current_key_index + 1}")
                        logger.error(f"Status code found in error: {e}")
                        
                        # If it's a 400 error, it might be a model/request issue, not auth
                        if "400" in error_msg:
                            logger.error("400 Bad Request - This might be a model compatibility issue, not auth")
                        
                        self.failed_keys.add(self.current_key_index)
                        
                        if not self._rotate_api_key():
                            return None
                        keys_tried += 1
                        break  # Try new key immediately
                    
                    # Other errors - retry with backoff
                    else:
                        logger.error(f"Error with key #{self.current_key_index + 1}: {e}")
                        
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.info(f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            # After max retries, try rotating key
                            logger.warning("Max retries reached, trying next key...")
                            if not self._rotate_api_key():
                                return None
                            keys_tried += 1
                            break
        
        logger.error("All API keys and retries exhausted")
        return None
    
    def _format_tables_for_llm(self, tables_list: List[Dict]) -> str:
        """
        Format tables as markdown for LLM input.
        
        Args:
            tables_list: List of table dictionaries
            
        Returns:
            Markdown-formatted tables
        """
        if not tables_list:
            return "No tables found on this page."
        
        formatted = []
        for i, table in enumerate(tables_list):
            formatted.append(f"\n### Table {i+1}\n")
            try:
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                if headers and rows:
                    # Create markdown table
                    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
                    separator = "| " + " | ".join("---" for _ in headers) + " |"
                    data_rows = []
                    for row in rows:
                        if len(row) == len(headers):
                            data_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
                    
                    formatted.append(header_row)
                    formatted.append(separator)
                    formatted.extend(data_rows)
            except Exception as e:
                logger.warning(f"Failed to format table {i}: {e}")
        
        return "\n".join(formatted)
    
    def extract_syndrome_data(
        self,
        syndrome_text: str,
        tables_list: List[Dict],
        page_number: int,
        source_pdf: str
    ) -> Optional[SyndromeData]:
        """
        Extract structured syndrome data using LLM.
        
        Args:
            syndrome_text: Text content of the syndrome section
            tables_list: List of tables from the section
            page_number: Source page number
            source_pdf: Source PDF filename
            
        Returns:
            SyndromeData object or None if extraction fails
        """
        tables_markdown = self._format_tables_for_llm(tables_list)
        
        # Create detailed extraction prompt
        prompt = f"""You are a medical data extraction expert. Extract structured information from ICMR antimicrobial treatment guidelines.

**CRITICAL INSTRUCTIONS:**
1. ONLY extract information EXPLICITLY stated in the text
2. Do NOT infer or add information not present
3. Pay attention to units (mg, g, kg, mL)
4. Distinguish between "first-line" and "alternative" therapies
5. Capture exact duration (e.g., "7-10 days", "until fever resolves for 48 hours")
6. Note if dosage is weight-based (mg/kg) or fixed

**TEXT TO ANALYZE:**

{syndrome_text[:8000]}

**TABLES:**

{tables_markdown}

**EXTRACT THE FOLLOWING STRUCTURED DATA:**

**REQUIRED FIELDS (MUST have a value, use descriptive text if not explicitly stated):**
- syndrome_name: Official syndrome name from ICMR (REQUIRED - extract the main heading or section title)
- definition: Clinical definition of the syndrome (REQUIRED - extract description, clinical presentation, or any defining information)
- common_pathogens: List of pathogens (can be empty array [] if none mentioned)
- first_line_drugs: List of first-line drugs (can be empty array [] if none mentioned)

**OPTIONAL FIELDS (can be null or empty):**
- icd10_codes: List of ICD-10 codes if mentioned (use [] if not mentioned)
- diagnostic_criteria: How to diagnose this condition (null if not stated)

common_pathogens: List of pathogens, each with:
  - organism_name: Scientific name
  - common_name: Lay term if different (optional)
  - prevalence: most common/less common/rare (optional)
  - resistance_pattern: ESBL/MRSA/carbapenem-resistant etc (optional)

first_line_drugs: List of first-line drugs, each with:
  - drug_name: GENERIC NAME IN UPPERCASE (REQUIRED)
  - brand_names: Commercial names if mentioned (optional)
  - dosage: Exact dosage with units (REQUIRED, use "Not specified" if truly missing)
  - route: IV/PO/IM/SC (REQUIRED, use "Not specified" if truly missing)
  - frequency: TID/BID/QID/once daily (REQUIRED, use "Not specified" if truly missing)
  - duration: Exact duration (REQUIRED, use "Not specified" if truly missing)
  - indication: first-line/alternative/severe cases (REQUIRED)
  - special_instructions: Any special notes (optional)

alternative_drugs: List of alternative drugs with same structure as first_line_drugs (optional)

special_populations: List of special population considerations (optional), each with:
  - population_type: pregnancy/pediatric/geriatric/renal impairment/hepatic impairment
  - modification: Dosage adjustment or alternative drug
  - trimester: first/second/third (for pregnancy, optional)
  - age_group: neonate/infant/child (for pediatrics, optional)

contraindications: List of contraindications (optional), each with:
  - condition: Contraindicated condition
  - severity: absolute/relative
  - reason: Explanation (optional)

monitoring_requirements: List of lab tests needed (optional)
de_escalation_guidance: When and how to de-escalate therapy (optional)

**IMPORTANT FORMATTING RULES:**
- Return ONLY a valid JSON object, no other text
- Extract information from the text provided
- **syndrome_name**: MUST have a value - use the section heading, title, or main syndrome name
- **definition**: MUST have a value - extract ANY descriptive text about the syndrome/infection/condition
- For optional LIST fields: use empty array [] if not found (NEVER use null for arrays)
- For optional STRING fields: use null if not found
- For drug fields (drug_name, dosage, route, etc.): use "Not specified" if missing but drug is mentioned
- Do NOT include "source_page" or "source_document" - these are added automatically

**CRITICAL: Never return null for syndrome_name or definition. Always extract the heading/title and some descriptive text.**
"""

        logger.info(f"Extracting data from page {page_number}...")
        
        # Call LLM with structured output and automatic key rotation
        data_dict = self._call_structured_llm_with_retry(prompt, max_retries=3)
        
        if not data_dict:
            logger.error("Failed to get structured response from LLM after all retries and key rotations")
            return None
        
        try:
            # Add source metadata
            data_dict['source_page'] = page_number
            data_dict['source_document'] = source_pdf
            
            # Validate and create final Pydantic model
            syndrome_data = SyndromeData(**data_dict)
            
            logger.info(f"✓ Successfully extracted: {syndrome_data.syndrome_name}")
            return syndrome_data
            
        except ValidationError as e:
            logger.error(f"Final validation error: {e}")
            logger.error(f"Data dict: {json.dumps(data_dict, indent=2)[:500]}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error creating SyndromeData: {e}")
            return None
    
    def _extract_single_syndrome(
        self,
        section: Dict[str, Any],
        index: int
    ) -> tuple[int, Optional[SyndromeData], Optional[str]]:
        """
        Extract a single syndrome (used for parallel processing).
        
        Args:
            section: Syndrome section data
            index: Index of the syndrome in the list
            
        Returns:
            Tuple of (index, syndrome_data, error_message)
        """
        try:
            syndrome_data = self.extract_syndrome_data(
                syndrome_text=section['full_text'],
                tables_list=section['tables'],
                page_number=section['page_start'],
                source_pdf=section.get('source_document', 'unknown')
            )
            
            if syndrome_data:
                return (index, syndrome_data, None)
            else:
                return (index, None, "Extraction returned None")
                
        except Exception as e:
            error_msg = f"Failed: {str(e)}"
            logger.error(f"Syndrome {section['syndrome_name']}: {error_msg}")
            return (index, None, error_msg)
    
    def batch_extract_syndromes(
        self,
        syndrome_sections_list: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        max_workers: int = None
    ) -> List[SyndromeData]:
        """
        Extract data from multiple syndrome sections using parallel processing.
        
        Args:
            syndrome_sections_list: List of syndrome sections from PDF parser
            output_dir: Directory to save intermediate results
            max_workers: Number of parallel workers (default: min(8, number of API keys))
            
        Returns:
            List of successfully extracted SyndromeData objects
        """
        from tqdm import tqdm
        
        # Default to number of API keys or 8, whichever is smaller
        if max_workers is None:
            max_workers = min(len(self.api_keys), 8)
        
        logger.info(f"Starting PARALLEL extraction of {len(syndrome_sections_list)} syndromes")
        logger.info(f"Using {max_workers} parallel workers")
        
        extracted_data = []
        failed_extractions = []
        results = {}  # Store results with their original index
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_section = {
                executor.submit(self._extract_single_syndrome, section, i): (i, section)
                for i, section in enumerate(syndrome_sections_list)
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(syndrome_sections_list), desc="Extracting syndromes") as pbar:
                for future in as_completed(future_to_section):
                    i, section = future_to_section[future]
                    
                    try:
                        index, syndrome_data, error = future.result()
                        
                        if syndrome_data:
                            results[index] = syndrome_data
                            pbar.set_postfix_str(f"✓ {syndrome_data.syndrome_name[:40]}")
                        else:
                            failed_extractions.append((section['syndrome_name'], error))
                            pbar.set_postfix_str(f"✗ Failed")
                        
                    except Exception as e:
                        logger.error(f"Exception in future: {e}")
                        failed_extractions.append((section['syndrome_name'], str(e)))
                    
                    pbar.update(1)
                    
                    # Save intermediate results every 5 completed extractions
                    if output_dir and len(results) % 5 == 0:
                        temp_data = [results[i] for i in sorted(results.keys())]
                        self._save_intermediate(temp_data, output_dir, len(results))
        
        # Sort results by original index to maintain order
        extracted_data = [results[i] for i in sorted(results.keys())]
        
        logger.info(f"Parallel extraction complete: {len(extracted_data)} successful, {len(failed_extractions)} failed")
        if failed_extractions:
            logger.warning(f"Failed syndromes ({len(failed_extractions)}):")
            for name, error in failed_extractions[:10]:  # Show first 10
                logger.warning(f"  - {name}: {error}")
        
        return extracted_data
    
    def _save_intermediate(self, data_list: List[SyndromeData], output_dir: str, batch_num: int) -> None:
        """Save intermediate results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            intermediate_file = output_path / f"intermediate_batch_{batch_num}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [s.model_dump() for s in data_list],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"Saved intermediate results to {intermediate_file}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def save_structured_data(self, syndrome_data_list: List[SyndromeData], output_dir: str) -> None:
        """
        Save structured data to files.
        
        Args:
            syndrome_data_list: List of SyndromeData objects
            output_dir: Directory to save output
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save all syndromes as single JSON
        all_syndromes_file = output_path / "all_syndromes.json"
        with open(all_syndromes_file, 'w', encoding='utf-8') as f:
            json.dump(
                [s.model_dump() for s in syndrome_data_list],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"Saved all syndromes to {all_syndromes_file}")
        
        # Save individual syndrome files
        for syndrome in syndrome_data_list:
            # Clean filename
            filename = syndrome.syndrome_name.replace('/', '_').replace('\\', '_')
            filename = ''.join(c for c in filename if c.isalnum() or c in (' ', '_', '-'))
            filename = filename[:100]  # Limit length
            
            syndrome_file = output_path / f"{filename}.json"
            with open(syndrome_file, 'w', encoding='utf-8') as f:
                json.dump(syndrome.model_dump(), f, indent=2, ensure_ascii=False)
        
        # Create summary CSV
        summary_data = []
        for syndrome in syndrome_data_list:
            summary_data.append({
                'syndrome_name': syndrome.syndrome_name,
                'number_of_pathogens': len(syndrome.common_pathogens),
                'number_of_first_line_drugs': len(syndrome.first_line_drugs),
                'number_of_alternatives': len(syndrome.alternative_drugs) if syndrome.alternative_drugs else 0,
                'page_number': syndrome.source_page
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / "summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")

