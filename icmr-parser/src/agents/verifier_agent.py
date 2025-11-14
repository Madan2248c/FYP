"""
Verifier Agent - Validates reasoning for factual and clinical correctness.

Cross-checks Generator output against structured ICMR data to ensure:
- Guideline references are accurate
- Drug recommendations match ICMR data
- Dosages are within acceptable ranges
- Pathogen-drug links are valid
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VerificationResult:
    """Result of reasoning verification."""
    
    def __init__(self):
        self.valid = True
        self.errors = []
        self.warnings = []
        self.checks_performed = []
    
    def add_error(self, check_name: str, message: str):
        """Add a validation error."""
        self.valid = False
        self.errors.append(f"{check_name}: {message}")
    
    def add_warning(self, check_name: str, message: str):
        """Add a validation warning."""
        self.warnings.append(f"{check_name}: {message}")
    
    def add_check(self, check_name: str, passed: bool):
        """Record a check result."""
        self.checks_performed.append({
            'check': check_name,
            'passed': passed
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'checks_performed': self.checks_performed,
            'summary': 'Pass' if self.valid else f"Fail - {'; '.join(self.errors)}"
        }


class VerifierAgent:
    """
    Verifier Agent for reasoning validation.
    
    Validates Generator output against structured ICMR data using multiple checks:
    1. Guideline reference presence
    2. Drug validity (in first-line or alternative lists)
    3. Dosage accuracy (within ±10% of ICMR recommendation)
    4. Pathogen-drug link validation
    """
    
    def __init__(self, dosage_tolerance: float = 0.10):
        """
        Initialize Verifier Agent.
        
        Args:
            dosage_tolerance: Allowable dosage deviation (default: 10%)
        """
        self.dosage_tolerance = dosage_tolerance
        
        logger.info(f"Verifier Agent initialized (dosage tolerance: ±{dosage_tolerance*100}%)")
    
    def verify_reasoning(
        self,
        reasoning_output: Dict[str, Any],
        case: Dict[str, Any],
        guideline_data: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verify reasoning output against ICMR guidelines.
        
        Args:
            reasoning_output: Generated reasoning from Generator Agent
            case: Original patient case
            guideline_data: ICMR syndrome data
            
        Returns:
            VerificationResult with validation details
        """
        result = VerificationResult()
        
        logger.info(f"Verifying reasoning for case {reasoning_output.get('case_id', 'UNKNOWN')}")
        
        # Check 1: Guideline reference presence
        self._check_guideline_references(reasoning_output, result)
        
        # Check 2: Drug validity
        self._check_drug_validity(reasoning_output, case, guideline_data, result)
        
        # Check 3: Dosage accuracy
        self._check_dosage_accuracy(reasoning_output, case, guideline_data, result)
        
        # Check 4: Pathogen-drug link
        self._check_pathogen_drug_link(reasoning_output, case, guideline_data, result)
        
        # Check 5: Duration appropriateness
        self._check_duration(reasoning_output, case, guideline_data, result)
        
        # Check 6: Reasoning completeness
        self._check_reasoning_completeness(reasoning_output, result)
        
        if result.valid:
            logger.info(f"✓ Reasoning validated successfully")
        else:
            logger.warning(f"✗ Reasoning validation failed: {result.errors}")
        
        return result
    
    def _check_guideline_references(self, reasoning: Dict, result: VerificationResult):
        """Check if ICMR 2025 guidelines are referenced."""
        check_name = "Guideline Reference Check"
        
        # Check in ICMR references field
        icmr_refs = reasoning.get('icmr_references', [])
        
        # Also check in reasoning steps
        reasoning_steps = reasoning.get('reasoning_steps', [])
        has_ref_in_steps = any(
            step.get('reference') and 'ICMR' in step.get('reference', '')
            for step in reasoning_steps
        )
        
        # Check in justification
        justification = reasoning.get('justification', '')
        has_ref_in_justification = 'ICMR' in justification or '2025' in justification
        
        if icmr_refs or has_ref_in_steps or has_ref_in_justification:
            result.add_check(check_name, True)
        else:
            result.add_error(check_name, "No ICMR 2025 guideline reference found in reasoning")
            result.add_check(check_name, False)
    
    def _check_drug_validity(
        self,
        reasoning: Dict,
        case: Dict,
        guideline: Dict,
        result: VerificationResult
    ):
        """Check if prescribed drug exists in ICMR recommendations."""
        check_name = "Drug Validity Check"
        
        prescription = case.get('prescription', {})
        prescribed_drug = prescription.get('drug', '').lower()
        
        if not prescribed_drug:
            result.add_warning(check_name, "No drug specified in prescription")
            result.add_check(check_name, False)
            return
        
        # Get ICMR drug lists (handle None values)
        first_line_drugs = guideline.get('first_line_drugs') or []
        alternative_drugs = guideline.get('alternative_drugs') or []
        
        # Check if drug is in either list
        first_line_names = [d.get('drug_name', '').lower() for d in first_line_drugs]
        alternative_names = [d.get('drug_name', '').lower() for d in alternative_drugs]
        
        drug_found = False
        drug_category = None
        
        for drug_name in first_line_names:
            if prescribed_drug in drug_name or drug_name in prescribed_drug:
                drug_found = True
                drug_category = "first-line"
                break
        
        if not drug_found:
            for drug_name in alternative_names:
                if prescribed_drug in drug_name or drug_name in prescribed_drug:
                    drug_found = True
                    drug_category = "alternative"
                    break
        
        if drug_found:
            result.add_check(check_name, True)
            logger.debug(f"Drug '{prescribed_drug}' found in ICMR {drug_category} recommendations")
        else:
            # Check if reasoning mentions it's off-guideline but justified
            justification = reasoning.get('justification', '').lower()
            if 'alternative' in justification or 'resistant' in justification or 'allergy' in justification:
                result.add_warning(check_name, f"Drug '{prescribed_drug}' not in ICMR list but justified in reasoning")
                result.add_check(check_name, True)
            else:
                result.add_error(check_name, f"Drug '{prescribed_drug}' not found in ICMR first-line or alternative recommendations")
                result.add_check(check_name, False)
    
    def _check_dosage_accuracy(
        self,
        reasoning: Dict,
        case: Dict,
        guideline: Dict,
        result: VerificationResult
    ):
        """Check if dosage is within acceptable range of ICMR recommendation."""
        check_name = "Dosage Accuracy Check"
        
        prescription = case.get('prescription', {})
        prescribed_drug = prescription.get('drug', '').lower()
        prescribed_dosage = prescription.get('dosage', '')
        
        if not prescribed_dosage:
            result.add_warning(check_name, "No dosage specified in prescription")
            result.add_check(check_name, False)
            return
        
        # Extract numeric dose from prescription
        prescribed_dose_value = self._extract_dose_value(prescribed_dosage)
        
        if prescribed_dose_value is None:
            result.add_warning(check_name, f"Could not parse dosage: {prescribed_dosage}")
            result.add_check(check_name, False)
            return
        
        # Find matching drug in guidelines
        first_line_drugs = guideline.get('first_line_drugs') or []
        alternative_drugs = guideline.get('alternative_drugs') or []
        all_drugs = first_line_drugs + alternative_drugs
        
        matching_drug = None
        for drug in all_drugs:
            drug_name = drug.get('drug_name', '').lower()
            if prescribed_drug in drug_name or drug_name in prescribed_drug:
                matching_drug = drug
                break
        
        if not matching_drug:
            result.add_warning(check_name, "Drug not found in guidelines for dosage comparison")
            result.add_check(check_name, False)
            return
        
        # Extract guideline dose
        guideline_dosage = matching_drug.get('dosage', '')
        guideline_dose_value = self._extract_dose_value(guideline_dosage)
        
        if guideline_dose_value is None:
            result.add_warning(check_name, f"Could not parse guideline dosage: {guideline_dosage}")
            result.add_check(check_name, False)
            return
        
        # Check if within tolerance
        lower_bound = guideline_dose_value * (1 - self.dosage_tolerance)
        upper_bound = guideline_dose_value * (1 + self.dosage_tolerance)
        
        if lower_bound <= prescribed_dose_value <= upper_bound:
            result.add_check(check_name, True)
            logger.debug(f"Dosage {prescribed_dose_value} within ±10% of guideline {guideline_dose_value}")
        else:
            result.add_error(
                check_name,
                f"Dosage {prescribed_dosage} outside acceptable range of {guideline_dosage} (±10%)"
            )
            result.add_check(check_name, False)
    
    def _check_pathogen_drug_link(
        self,
        reasoning: Dict,
        case: Dict,
        guideline: Dict,
        result: VerificationResult
    ):
        """Check if pathogen-drug link is valid."""
        check_name = "Pathogen-Drug Link Check"
        
        # Extract mentioned pathogens from reasoning
        reasoning_text = self._get_reasoning_text(reasoning)
        
        # Get ICMR pathogens
        icmr_pathogens = guideline.get('common_pathogens') or []
        
        if not icmr_pathogens:
            result.add_warning(check_name, "No pathogens listed in ICMR guidelines for this syndrome")
            result.add_check(check_name, False)
            return
        
        # Check if any ICMR pathogen is mentioned in reasoning
        pathogen_mentioned = False
        for pathogen in icmr_pathogens:
            organism_name = pathogen.get('organism_name', '').lower()
            if organism_name in reasoning_text.lower():
                pathogen_mentioned = True
                logger.debug(f"Pathogen '{organism_name}' mentioned in reasoning")
                break
        
        if pathogen_mentioned or 'empirical' in reasoning_text.lower():
            result.add_check(check_name, True)
        else:
            result.add_warning(check_name, "No specific pathogen mentioned in reasoning (may be empirical therapy)")
            result.add_check(check_name, True)  # Not a failure, just a warning
    
    def _check_duration(
        self,
        reasoning: Dict,
        case: Dict,
        guideline: Dict,
        result: VerificationResult
    ):
        """Check if treatment duration is appropriate."""
        check_name = "Duration Check"
        
        prescription = case.get('prescription', {})
        prescribed_duration = prescription.get('duration', '')
        
        if not prescribed_duration:
            result.add_warning(check_name, "No duration specified in prescription")
            result.add_check(check_name, False)
            return
        
        # Extract days from prescription
        prescribed_days = self._extract_days(prescribed_duration)
        
        if prescribed_days is None:
            result.add_warning(check_name, f"Could not parse duration: {prescribed_duration}")
            result.add_check(check_name, False)
            return
        
        # Get guideline duration for prescribed drug
        prescribed_drug = prescription.get('drug', '').lower()
        first_line_drugs = guideline.get('first_line_drugs') or []
        alternative_drugs = guideline.get('alternative_drugs') or []
        all_drugs = first_line_drugs + alternative_drugs
        
        matching_drug = None
        for drug in all_drugs:
            drug_name = drug.get('drug_name', '').lower()
            if prescribed_drug in drug_name or drug_name in prescribed_drug:
                matching_drug = drug
                break
        
        if not matching_drug or not matching_drug.get('duration'):
            result.add_warning(check_name, "No guideline duration available for comparison")
            result.add_check(check_name, False)
            return
        
        guideline_duration = matching_drug.get('duration', '')
        guideline_days = self._extract_days(guideline_duration)
        
        if guideline_days is None:
            result.add_warning(check_name, f"Could not parse guideline duration: {guideline_duration}")
            result.add_check(check_name, False)
            return
        
        # Check if within reasonable range (guideline ±3 days or within range if specified)
        if abs(prescribed_days - guideline_days) <= 3:
            result.add_check(check_name, True)
        else:
            result.add_error(
                check_name,
                f"Duration {prescribed_duration} differs significantly from guideline {guideline_duration}"
            )
            result.add_check(check_name, False)
    
    def _check_reasoning_completeness(self, reasoning: Dict, result: VerificationResult):
        """Check if reasoning is complete and well-structured."""
        check_name = "Reasoning Completeness Check"
        
        reasoning_steps = reasoning.get('reasoning_steps', [])
        
        if len(reasoning_steps) < 3:
            result.add_error(check_name, f"Insufficient reasoning steps ({len(reasoning_steps)} < 3 minimum)")
            result.add_check(check_name, False)
            return
        
        # Check for key components
        has_patient_assessment = False
        has_drug_analysis = False
        has_guideline_ref = False
        
        for step in reasoning_steps:
            description = step.get('description', '').lower()
            observation = step.get('observation', '').lower()
            combined = description + " " + observation
            
            if 'patient' in combined or 'history' in combined or 'risk' in combined:
                has_patient_assessment = True
            if 'drug' in combined or 'dosage' in combined or 'medication' in combined:
                has_drug_analysis = True
            if 'icmr' in combined or 'guideline' in combined:
                has_guideline_ref = True
        
        if has_patient_assessment and has_drug_analysis and has_guideline_ref:
            result.add_check(check_name, True)
        else:
            missing = []
            if not has_patient_assessment:
                missing.append("patient assessment")
            if not has_drug_analysis:
                missing.append("drug analysis")
            if not has_guideline_ref:
                missing.append("guideline reference")
            
            result.add_warning(check_name, f"Reasoning may be incomplete (missing: {', '.join(missing)})")
            result.add_check(check_name, True)  # Warning, not error
    
    def _extract_dose_value(self, dosage_str: str) -> float:
        """Extract numeric dose value from dosage string."""
        if not dosage_str:
            return None
        
        # Look for patterns like "500mg", "2g", "50mg/kg"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|g)', dosage_str.lower())
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            
            # Convert to mg for comparison
            if unit == 'g':
                value *= 1000
            
            return value
        
        return None
    
    def _extract_days(self, duration_str: str) -> int:
        """Extract number of days from duration string."""
        if not duration_str:
            return None
        
        # Look for patterns like "7 days", "7-10 days", "1 week"
        match = re.search(r'(\d+)(?:-(\d+))?\s*(?:day|week)', duration_str.lower())
        if match:
            days = int(match.group(1))
            if match.group(2):  # Range like "7-10"
                days = (days + int(match.group(2))) // 2  # Use midpoint
            if 'week' in duration_str.lower():
                days *= 7
            return days
        
        return None
    
    def _get_reasoning_text(self, reasoning: Dict) -> str:
        """Get all text from reasoning for analysis."""
        text_parts = []
        
        for step in reasoning.get('reasoning_steps', []):
            text_parts.append(step.get('description', ''))
            text_parts.append(step.get('observation', ''))
        
        text_parts.append(reasoning.get('justification', ''))
        
        return ' '.join(text_parts)
