"""
Evaluation Metrics for AMR Prescription Validation GRPO Training.

This module defines the evaluation criteria for assessing model-generated
prescription validations against ICMR 2025 guidelines.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Defines all evaluation metrics for prescription validation quality."""
    
    # Core metrics for prescription validation
    CLINICAL_ACCURACY = "clinical_accuracy"  # Correctness of medical reasoning
    GUIDELINE_ADHERENCE = "guideline_adherence"  # Alignment with ICMR 2025
    REASONING_COMPLETENESS = "reasoning_completeness"  # All steps covered
    SAFETY_AWARENESS = "safety_awareness"  # Identifies risks/contraindications
    DECISION_APPROPRIATENESS = "decision_appropriateness"  # Approve/Modify/Reject correctness
    REFERENCE_ACCURACY = "reference_accuracy"  # Correct ICMR page citations


# Evaluation criteria definitions
EVALUATION_CRITERIA = {
    "clinical_accuracy": {
        "description": "Correctness of clinical reasoning and medical facts",
        "scale": "1-5 (1=Incorrect medical facts, 5=Perfectly accurate)",
        "weight": 0.25,
        "examples": {
            "5": "All medical facts, drug dosages, and pathogen information are correct",
            "3": "Mostly correct but minor inaccuracies in dosing or pathogen details",
            "1": "Major medical errors or incorrect drug recommendations"
        }
    },
    "guideline_adherence": {
        "description": "Alignment with ICMR 2025 antimicrobial treatment guidelines",
        "scale": "1-5 (1=Contradicts guidelines, 5=Perfect adherence)",
        "weight": 0.25,
        "examples": {
            "5": "Perfectly follows ICMR first-line recommendations",
            "3": "Generally follows guidelines but misses some nuances",
            "1": "Recommends drugs not in ICMR guidelines or contradicts them"
        }
    },
    "reasoning_completeness": {
        "description": "Coverage of all required reasoning steps (patient assessment, diagnosis, drug selection, dosage, duration)",
        "scale": "1-5 (1=Missing critical steps, 5=All steps covered)",
        "weight": 0.20,
        "examples": {
            "5": "Covers patient assessment, diagnosis validation, pathogen ID, drug selection, dosage verification, duration assessment",
            "3": "Covers most steps but misses 1-2 important considerations",
            "1": "Skips multiple critical reasoning steps"
        }
    },
    "safety_awareness": {
        "description": "Identification of allergies, contraindications, drug interactions, and special populations",
        "scale": "1-5 (1=Ignores safety, 5=Comprehensive safety analysis)",
        "weight": 0.15,
        "examples": {
            "5": "Identifies all relevant allergies, contraindications, renal dosing needs",
            "3": "Mentions some safety concerns but misses important ones",
            "1": "Fails to identify critical safety issues"
        }
    },
    "decision_appropriateness": {
        "description": "Correctness of final decision (Approve/Modify/Reject) based on the case",
        "scale": "1-5 (1=Wrong decision, 5=Perfect decision)",
        "weight": 0.10,
        "examples": {
            "5": "Decision perfectly matches what ICMR guidelines would recommend",
            "3": "Decision is reasonable but could be more precise",
            "1": "Decision contradicts clear guideline recommendations"
        }
    },
    "reference_accuracy": {
        "description": "Accuracy of ICMR guideline page citations and references",
        "scale": "1-5 (1=No/wrong citations, 5=Perfect citations)",
        "weight": 0.05,
        "examples": {
            "5": "Cites correct ICMR page numbers and guideline sections",
            "3": "Mentions ICMR but page numbers may be approximate",
            "1": "No citations or completely incorrect references"
        }
    }
}


# Priority metrics for faster training (top 3 most important)
PRIORITY_METRICS = [
    "clinical_accuracy",
    "guideline_adherence",
    "reasoning_completeness"
]

# All metrics for comprehensive evaluation
ALL_METRICS = list(EVALUATION_CRITERIA.keys())


def get_metric_weights() -> Dict[str, float]:
    """Get normalized weights for all metrics."""
    return {metric: criteria["weight"] for metric, criteria in EVALUATION_CRITERIA.items()}


def validate_weights():
    """Ensure all weights sum to 1.0."""
    total = sum(get_metric_weights().values())
    assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"
    print(f"✅ Metric weights validated: {total:.3f}")


if __name__ == "__main__":
    print("=" * 80)
    print("AMR Prescription Validation - Evaluation Metrics")
    print("=" * 80)
    print()
    
    for metric, criteria in EVALUATION_CRITERIA.items():
        print(f"📊 {metric.upper()}")
        print(f"   Description: {criteria['description']}")
        print(f"   Scale: {criteria['scale']}")
        print(f"   Weight: {criteria['weight']} ({criteria['weight']*100:.0f}%)")
        print()
    
    validate_weights()
    
    print(f"\n🎯 Priority Metrics (for faster training): {PRIORITY_METRICS}")
    print(f"📋 All Metrics (for comprehensive evaluation): {ALL_METRICS}")

