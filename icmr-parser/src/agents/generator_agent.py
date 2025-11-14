"""
Generator Agent - Produces step-by-step clinical reasoning for prescription validation.

Uses Gemini 1.5 Pro with structured output to generate chain-of-thought reasoning
that validates prescriptions against ICMR 2025 guidelines.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReasoningStep(BaseModel):
    """Single step in the reasoning chain."""
    step_number: int = Field(description="Sequential step number")
    description: str = Field(description="What is being analyzed in this step")
    observation: str = Field(description="Clinical observation or finding")
    reference: Optional[str] = Field(default=None, description="ICMR guideline reference if applicable")


class GeneratorOutput(BaseModel):
    """Structured output from Generator Agent."""
    case_id: str = Field(description="Unique case identifier")
    reasoning_steps: List[ReasoningStep] = Field(description="Step-by-step reasoning chain")
    final_decision: str = Field(description="Approved / Modify / Reject")
    justification: str = Field(description="Detailed explanation of the decision")
    icmr_references: List[str] = Field(description="List of ICMR guideline references cited")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")


class GeneratorAgent:
    """
    Generator Agent for clinical reasoning.
    
    Produces step-by-step chain-of-thought reasoning for prescription validation
    using Gemini 1.5 Pro with structured output.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro", thinking_budget: int = -1):
        """
        Initialize Generator Agent.
        
        Args:
            api_key: Gemini API key
            model_name: Model to use (default: gemini-2.5-pro with thinking)
            thinking_budget: Token budget for thinking (-1 for dynamic, 128-32768 for fixed)
                            See: https://ai.google.dev/gemini-api/docs/thinking
        """
        self.api_key = api_key
        self.model_name = model_name
        self.thinking_budget = thinking_budget
        self.client = genai.Client(api_key=api_key)
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"Generator Agent initialized with {model_name}")
        logger.info(f"Thinking mode: {'Dynamic' if thinking_budget == -1 else f'{thinking_budget} tokens'}")
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        template_path = Path(__file__).parent.parent / "prompts" / "generator_prompt.txt"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()
        else:
            # Default template if file doesn't exist
            return """You are a clinical pharmacist with expertise in antimicrobial stewardship, reasoning step-by-step to validate prescriptions according to ICMR 2025 Antimicrobial Treatment Guidelines.

Your task is to analyze the patient case and prescription, then provide detailed clinical reasoning.

**REASONING FRAMEWORK:**

1. **Patient Assessment**
   - Review patient profile (age, sex, comorbidities)
   - Identify risk factors and contraindications
   - Consider special populations (pregnancy, pediatric, renal/hepatic impairment)

2. **Diagnosis Validation**
   - Confirm diagnosis matches clinical presentation
   - Check if symptoms align with suspected pathogen
   - Reference ICMR 2025 diagnostic criteria

3. **Pathogen Identification**
   - Identify likely causative organisms from ICMR data
   - Consider local resistance patterns (ESBL, MRSA, CRE)
   - Assess empirical vs targeted therapy needs

4. **Drug Selection Analysis**
   - Verify if prescribed drug is in ICMR first-line recommendations
   - Check for contraindications or drug interactions
   - Consider spectrum of activity vs likely pathogen

5. **Dosage & Route Verification**
   - Compare prescribed dose to ICMR recommended dosage
   - Verify route of administration (IV/PO) is appropriate
   - Check frequency matches guideline (TID, BID, QID)

6. **Duration Assessment**
   - Evaluate if treatment duration meets ICMR recommendations
   - Consider if extension needed for complications
   - Reference guideline-specific duration (e.g., "7-10 days")

7. **Final Decision**
   - Approved: Prescription aligns with ICMR 2025 guidelines
   - Modify: Needs adjustment (specify what to change)
   - Reject: Inappropriate choice (explain why and suggest alternative)

**CRITICAL REQUIREMENTS:**
- MUST cite specific ICMR 2025 guideline sections or page numbers
- MUST reference the syndrome-specific recommendations
- MUST provide step-by-step reasoning before final decision
- MUST justify any recommended modifications
- Use clinical pharmacology principles throughout

Provide structured, evidence-based reasoning that demonstrates deep understanding of antimicrobial therapy principles."""
    
    def generate_reasoning(
        self,
        case: Dict[str, Any],
        guideline_data: Dict[str, Any]
    ) -> Optional[GeneratorOutput]:
        """
        Generate step-by-step clinical reasoning for a case.
        
        Args:
            case: Patient case dictionary with diagnosis, prescription, etc.
            guideline_data: ICMR syndrome data for reference
            
        Returns:
            GeneratorOutput with structured reasoning or None if failed
        """
        case_id = case.get('case_id', 'UNKNOWN')
        logger.info(f"Generating reasoning for case {case_id}")
        
        # Build context from case and guidelines
        context = self._build_context(case, guideline_data)
        
        # Create full prompt
        prompt = f"""{self.prompt_template}

**PATIENT CASE:**
{self._format_case(case)}

**ICMR 2025 GUIDELINE DATA:**
{self._format_guideline_data(guideline_data)}

**YOUR TASK:**
Analyze this case step-by-step and provide your clinical reasoning. For each step, explain:
1. What you're analyzing
2. Your observation
3. Reference to ICMR 2025 guideline when applicable

Then provide your final decision (Approved/Modify/Reject) with detailed justification.

**IMPORTANT:** Use case_id = "{case_id}" in your response.
"""
        
        try:
            # Call Gemini 2.5 Pro with thinking + structured output
            # See: https://ai.google.dev/gemini-api/docs/thinking
            # See: https://ai.google.dev/gemini-api/docs/structured-output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=GeneratorOutput,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    )
                )
            )
            
            # Get parsed output
            reasoning_output: GeneratorOutput = response.parsed
            
            if reasoning_output is None:
                logger.warning(f"Gemini returned None for case {case_id}")
                return None
            
            # Override the case_id to use the actual input case_id
            # (LLM sometimes generates its own random IDs)
            reasoning_output.case_id = case_id
            
            # Log thinking tokens usage
            if hasattr(response, 'usage_metadata'):
                thoughts_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
                logger.debug(f"Thinking tokens: {thoughts_tokens}, Output tokens: {output_tokens}")
            
            logger.info(f"✓ Generated {len(reasoning_output.reasoning_steps)} reasoning steps for {case_id}")
            return reasoning_output
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning for {case_id}: {e}")
            return None
    
    def _build_context(self, case: Dict, guideline_data: Dict) -> str:
        """Build combined context from case and guidelines."""
        return f"Case: {case.get('diagnosis')}\nGuideline: {guideline_data.get('syndrome_name')}"
    
    def _format_case(self, case: Dict) -> str:
        """Format case data for prompt."""
        lines = []
        
        # Patient profile
        profile = case.get('patient_profile', {})
        lines.append(f"**Patient:** {profile.get('age')}yo {profile.get('sex', 'Unknown sex')}")
        
        if 'history' in profile and profile['history']:
            lines.append(f"**Medical History:** {', '.join(profile['history'])}")
        
        # Diagnosis and symptoms
        lines.append(f"**Diagnosis:** {case.get('diagnosis', 'Not specified')}")
        
        symptoms = case.get('symptoms', [])
        if symptoms:
            lines.append(f"**Presenting Symptoms:** {', '.join(symptoms)}")
        
        # Prescription
        prescription = case.get('prescription', {})
        lines.append(f"\n**PRESCRIPTION TO VALIDATE:**")
        lines.append(f"- Drug: {prescription.get('drug', 'Unknown')}")
        lines.append(f"- Dosage: {prescription.get('dosage', 'Unknown')}")
        lines.append(f"- Route: {prescription.get('route', 'Not specified')}")
        lines.append(f"- Duration: {prescription.get('duration', 'Not specified')}")
        
        return '\n'.join(lines)
    
    def _format_guideline_data(self, guideline: Dict) -> str:
        """Format ICMR guideline data for prompt."""
        lines = []
        
        lines.append(f"**Syndrome:** {guideline.get('syndrome_name', 'Unknown')}")
        lines.append(f"**Definition:** {guideline.get('definition', 'Not available')[:200]}...")
        
        # Common pathogens
        pathogens = guideline.get('common_pathogens', [])
        if pathogens:
            lines.append(f"\n**Common Pathogens (ICMR 2025):**")
            for i, pathogen in enumerate(pathogens[:5], 1):
                name = pathogen.get('organism_name', 'Unknown')
                resistance = pathogen.get('resistance_pattern')
                prevalence = pathogen.get('prevalence')
                
                path_line = f"{i}. {name}"
                if resistance:
                    path_line += f" ({resistance})"
                if prevalence:
                    path_line += f" - {prevalence}"
                
                lines.append(path_line)
        
        # First-line drugs
        first_line = guideline.get('first_line_drugs', [])
        if first_line:
            lines.append(f"\n**First-Line Drugs (ICMR 2025):**")
            for i, drug in enumerate(first_line[:5], 1):
                drug_name = drug.get('drug_name', 'Unknown')
                dosage = drug.get('dosage', 'dose not specified')
                route = drug.get('route', 'route not specified')
                duration = drug.get('duration', 'duration not specified')
                
                lines.append(f"{i}. {drug_name} - {dosage} {route} for {duration}")
        
        # Alternative drugs
        alternatives = guideline.get('alternative_drugs', [])
        if alternatives:
            lines.append(f"\n**Alternative Drugs (ICMR 2025):**")
            for i, drug in enumerate(alternatives[:3], 1):
                drug_name = drug.get('drug_name', 'Unknown')
                indication = drug.get('indication', '')
                lines.append(f"{i}. {drug_name} ({indication})")
        
        # Source reference
        page = guideline.get('source_page', 'unknown')
        lines.append(f"\n**ICMR Reference:** Page {page}, ICMR 2025 Guidelines")
        
        return '\n'.join(lines)

