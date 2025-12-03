"""
LLM-as-a-Judge Evaluation System using Groq Llama.

This module implements the evaluation API that scores model-generated
prescription validations against ground truth using multiple criteria.
"""

import os
import json
from typing import Dict, List, Optional
from groq import Groq
from evaluation_metrics import EVALUATION_CRITERIA, get_metric_weights


class LLMJudge:
    """
    LLM-as-a-Judge evaluator using Groq Llama for scoring prescription validations.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1
    ):
        """
        Initialize LLM Judge.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Groq model to use for evaluation
            temperature: Temperature for generation (low for consistency)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        
        print(f"✅ LLM Judge initialized with {model}")
    
    def _create_evaluation_prompt(
        self,
        metric: str,
        patient_case: Dict,
        model_output: str,
        ground_truth: str
    ) -> str:
        """
        Create evaluation prompt for a specific metric.
        
        Args:
            metric: Metric to evaluate (e.g., "clinical_accuracy")
            patient_case: Patient case context
            model_output: Model-generated prescription validation
            ground_truth: Reference/correct validation
            
        Returns:
            Formatted evaluation prompt
        """
        criteria = EVALUATION_CRITERIA[metric]
        
        prompt = f"""You are an expert medical evaluator specializing in antimicrobial stewardship and ICMR 2025 guidelines.

**EVALUATION TASK:**
Evaluate the quality of a model-generated prescription validation against a reference answer.

**METRIC TO EVALUATE:** {metric}
**Description:** {criteria['description']}
**Scale:** {criteria['scale']}

**EVALUATION EXAMPLES:**
{json.dumps(criteria['examples'], indent=2)}

**PATIENT CASE:**
```json
{json.dumps(patient_case, indent=2)}
```

**MODEL OUTPUT (to evaluate):**
{model_output}

**REFERENCE ANSWER (ground truth):**
{ground_truth}

**INSTRUCTIONS:**
1. Carefully compare the MODEL OUTPUT against the REFERENCE ANSWER
2. Evaluate ONLY the "{metric}" aspect based on the criteria above
3. Assign a score from 1-5 (integer only)
4. Provide a brief justification (2-3 sentences)

**OUTPUT FORMAT (JSON only):**
{{
  "metric": "{metric}",
  "score": <integer 1-5>,
  "justification": "<brief explanation>"
}}

Respond with ONLY the JSON object, no additional text.
"""
        return prompt
    
    def evaluate_single_metric(
        self,
        metric: str,
        patient_case: Dict,
        model_output: str,
        ground_truth: str
    ) -> Dict:
        """
        Evaluate a single metric for a model output.
        
        Args:
            metric: Metric name (e.g., "clinical_accuracy")
            patient_case: Patient case context
            model_output: Model-generated validation
            ground_truth: Reference validation
            
        Returns:
            Dict with metric, score, and justification
        """
        prompt = self._create_evaluation_prompt(
            metric, patient_case, model_output, ground_truth
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical evaluation expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate score is in range
            if not (1 <= result.get("score", 0) <= 5):
                raise ValueError(f"Score {result.get('score')} out of range [1-5]")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error evaluating {metric}: {e}")
            # Return default low score on error
            return {
                "metric": metric,
                "score": 1,
                "justification": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_all_metrics(
        self,
        patient_case: Dict,
        model_output: str,
        ground_truth: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate all specified metrics for a model output.
        
        Args:
            patient_case: Patient case context
            model_output: Model-generated validation
            ground_truth: Reference validation
            metrics: List of metrics to evaluate (None = all metrics)
            
        Returns:
            Dict mapping metric names to evaluation results
        """
        if metrics is None:
            metrics = list(EVALUATION_CRITERIA.keys())
        
        results = {}
        for metric in metrics:
            result = self.evaluate_single_metric(
                metric, patient_case, model_output, ground_truth
            )
            results[metric] = result
        
        return results
    
    def compute_weighted_reward(
        self,
        evaluation_results: Dict[str, Dict]
    ) -> float:
        """
        Compute weighted reward from evaluation results.
        
        Args:
            evaluation_results: Dict of metric evaluations
            
        Returns:
            Weighted reward score normalized to [0, 1]
        """
        weights = get_metric_weights()
        total_reward = 0.0
        total_weight = 0.0
        
        for metric, result in evaluation_results.items():
            if metric in weights:
                # Normalize score from 1-5 scale to 0-1 scale
                normalized_score = (result["score"] - 1) / 4.0
                total_reward += normalized_score * weights[metric]
                total_weight += weights[metric]
        
        # Normalize by total weight
        if total_weight > 0:
            return total_reward / total_weight
        return 0.0


# API endpoint handler function (for Supabase Edge Function)
def evaluate_endpoint(request_body: Dict) -> Dict:
    """
    Main API endpoint handler for evaluation requests.
    
    Expected request body:
    {
        "patient_case": {...},
        "model_output": "...",
        "ground_truth": "...",
        "metrics": ["clinical_accuracy", ...] (optional)
    }
    
    Returns:
    {
        "evaluations": {...},
        "weighted_reward": 0.85,
        "success": true
    }
    """
    try:
        # Extract request data
        patient_case = request_body.get("patient_case", {})
        model_output = request_body.get("model_output", "")
        ground_truth = request_body.get("ground_truth", "")
        metrics = request_body.get("metrics", None)
        
        # Validate inputs
        if not model_output or not ground_truth:
            return {
                "success": False,
                "error": "Missing required fields: model_output and ground_truth"
            }
        
        # Initialize judge
        judge = LLMJudge()
        
        # Evaluate
        evaluations = judge.evaluate_all_metrics(
            patient_case, model_output, ground_truth, metrics
        )
        
        # Compute weighted reward
        reward = judge.compute_weighted_reward(evaluations)
        
        return {
            "success": True,
            "evaluations": evaluations,
            "weighted_reward": reward,
            "model": judge.model
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the LLM Judge
    print("=" * 80)
    print("Testing LLM Judge")
    print("=" * 80)
    
    # Sample test case
    test_case = {
        "patient_profile": {
            "age": 45,
            "sex": "M",
            "history": ["diabetes"],
            "allergies": []
        },
        "diagnosis": "Community Acquired Pneumonia",
        "symptoms": ["fever", "cough", "dyspnea"],
        "prescription": {
            "drug": "PIPERACILLIN-TAZOBACTAM",
            "dosage": "4.5g",
            "route": "IV",
            "duration": "7 days"
        }
    }
    
    model_output = """
Step 1: Patient is 45-year-old male with diabetes. No contraindications noted.
Step 2: Diagnosis of CAP confirmed with typical symptoms.
Step 3: ICMR recommends Piperacillin-Tazobactam as first-line for CAP.
Step 4: Dosage of 4.5g IV matches ICMR guidelines exactly.
Step 5: Duration of 7 days aligns with recommendations.

Approved: Prescription is compliant with ICMR 2025 guidelines.
Reference: ICMR 2025, Page 108
"""
    
    ground_truth = """
Step 1: Patient Assessment - 45-year-old male with diabetes comorbidity.
Step 2: Diagnosis Validation - CAP confirmed with fever, cough, dyspnea.
Step 3: Pathogen Identification - Likely S. pneumoniae, H. influenzae.
Step 4: Drug Selection - Piperacillin-Tazobactam is ICMR first-line.
Step 5: Dosage Verification - 4.5g IV q6h matches guideline.
Step 6: Duration Assessment - 7 days per ICMR recommendation.

Approved: Fully compliant with ICMR 2025 Guidelines for CAP.
Reference: Page 108, ICMR 2025 Guidelines
"""
    
    # Initialize judge
    judge = LLMJudge()
    
    # Evaluate on priority metrics only (for speed)
    from evaluation_metrics import PRIORITY_METRICS
    
    print(f"\n🎯 Evaluating with priority metrics: {PRIORITY_METRICS}\n")
    
    results = judge.evaluate_all_metrics(
        test_case, model_output, ground_truth, PRIORITY_METRICS
    )
    
    print("\n📊 Evaluation Results:")
    print("=" * 80)
    for metric, result in results.items():
        print(f"\n{metric}:")
        print(f"  Score: {result['score']}/5")
        print(f"  Justification: {result['justification']}")
    
    reward = judge.compute_weighted_reward(results)
    print(f"\n💰 Weighted Reward: {reward:.4f}")
    print("=" * 80)

