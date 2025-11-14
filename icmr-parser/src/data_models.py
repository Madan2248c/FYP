"""
Pydantic models for structured medical data extraction from ICMR guidelines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DrugRecommendation(BaseModel):
    """Model for drug recommendation with complete dosing information."""
    
    drug_name: str = Field(description="Generic drug name")
    brand_names: Optional[List[str]] = Field(default=None, description="Commercial names if mentioned")
    dosage: Optional[str] = Field(default=None, description="Exact dosage with units like '500mg' or '1g'")
    route: Optional[str] = Field(default=None, description="Administration route: IV, PO, IM, SC")
    frequency: Optional[str] = Field(default=None, description="Dosing frequency: TID, BID, QID, once daily")
    duration: Optional[str] = Field(default=None, description="Treatment duration in days or as specified")
    indication: Optional[str] = Field(default=None, description="When to use: 'first-line', 'alternative', 'severe cases', 'resistant organisms'")
    special_instructions: Optional[str] = Field(default=None, description="Any special dosing notes")


class Pathogen(BaseModel):
    """Model for pathogen information."""
    
    organism_name: str = Field(description="Scientific name of pathogen")
    common_name: Optional[str] = Field(default=None, description="Lay term if different")
    prevalence: Optional[str] = Field(default=None, description="How common: 'most common', 'less common', 'rare'")
    resistance_pattern: Optional[str] = Field(default=None, description="AMR information like 'ESBL-producing', 'MRSA', 'carbapenem-resistant'")


class SpecialPopulation(BaseModel):
    """Model for special population considerations."""
    
    population_type: str = Field(description="'pregnancy', 'pediatric', 'geriatric', 'renal impairment', 'hepatic impairment'")
    modification: str = Field(description="Dosage adjustment or alternative drug")
    trimester: Optional[str] = Field(default=None, description="For pregnancy: 'first', 'second', 'third'")
    age_group: Optional[str] = Field(default=None, description="For pediatrics: 'neonate', 'infant', 'child'")


class Contraindication(BaseModel):
    """Model for contraindications."""
    
    condition: str = Field(description="Contraindicated condition")
    severity: str = Field(description="'absolute' or 'relative'")
    reason: Optional[str] = Field(default=None, description="Explanation")


class SyndromeData(BaseModel):
    """Main model for syndrome data structure."""
    
    syndrome_name: str = Field(description="Official syndrome name from ICMR")
    icd10_codes: Optional[List[str]] = Field(default=None, description="ICD-10 diagnostic codes")
    definition: str = Field(description="Clinical definition")
    diagnostic_criteria: Optional[str] = Field(default=None, description="How to diagnose")
    common_pathogens: List[Pathogen] = Field(description="List of common pathogens")
    first_line_drugs: List[DrugRecommendation] = Field(description="First-line drug recommendations")
    alternative_drugs: Optional[List[DrugRecommendation]] = Field(default=None, description="Alternative drug options")
    special_populations: Optional[List[SpecialPopulation]] = Field(default=None, description="Special population considerations")
    contraindications: Optional[List[Contraindication]] = Field(default=None, description="Contraindications")
    monitoring_requirements: Optional[List[str]] = Field(default=None, description="Lab tests needed")
    de_escalation_guidance: Optional[str] = Field(default=None, description="When and how to de-escalate")
    source_page: int = Field(description="PDF page number")
    source_document: str = Field(description="Which ICMR PDF this came from")

