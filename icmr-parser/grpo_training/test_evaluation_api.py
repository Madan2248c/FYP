#!/usr/bin/env python3
"""
Test script for the Supabase Edge Function evaluation API.

This script tests the deployed evaluation API to ensure it's working correctly.
"""

import os
import json
import requests
from pathlib import Path

def test_evaluation_api(api_url: str, groq_key: str):
    """Test the evaluation API with sample data."""

    print("🧪 Testing AMR Evaluation API")
    print("=" * 50)

    # Test payload
    test_payload = {
        "patient_case": {
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
        },
        "model_output": """Step 1: Patient Assessment - 45-year-old male with diabetes comorbidity.
Step 2: Diagnosis Validation - CAP confirmed with fever, cough, dyspnea symptoms.
Step 3: Pathogen Identification - Likely S. pneumoniae, H. influenzae.
Step 4: Drug Selection - Piperacillin-Tazobactam is first-line per ICMR guidelines.
Step 5: Dosage Verification - 4.5g IV matches ICMR recommendation.
Step 6: Duration Assessment - 7 days is appropriate.

Approved: Prescription fully compliant with ICMR 2025 guidelines.
Reference: ICMR 2025 Guidelines, Page 108""",
        "ground_truth": """Step 1: Patient Assessment - 45-year-old male with diabetes comorbidity.
Step 2: Diagnosis Validation - CAP confirmed with fever, cough, dyspnea symptoms.
Step 3: Pathogen Identification - Likely S. pneumoniae, H. influenzae.
Step 4: Drug Selection - Piperacillin-Tazobactam is ICMR first-line.
Step 5: Dosage Verification - 4.5g IV q6h matches guidelines exactly.
Step 6: Duration Assessment - 7 days per ICMR recommendation.

Approved: Fully compliant with ICMR 2025 Guidelines for CAP.
Reference: Page 108, ICMR 2025 Guidelines""",
        "metrics": ["clinical_accuracy", "guideline_adherence", "reasoning_completeness"]
    }

    print(f"📡 Testing API: {api_url}")
    print("📊 Metrics: clinical_accuracy, guideline_adherence, reasoning_completeness"
    print()

    try:
        # Make request
        print("🚀 Making API request...")
        response = requests.post(
            api_url,
            json=test_payload,
            timeout=60
        )

        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:"            print(json.dumps(result, indent=2))

            # Check if successful
            if result.get("success"):
                print("\n🎉 API Test PASSED!")
                print(f"📈 Weighted Reward: {result.get('weighted_reward', 0):.4f}")
                print(f"📋 Metrics Evaluated: {result.get('metrics_evaluated', 0)}")
                print(f"🤖 Model: {result.get('model', 'unknown')}")

                # Show individual scores
                evaluations = result.get("evaluations", {})
                print("\n📊 Individual Scores:")
                for metric, data in evaluations.items():
                    score = data.get("score", 0)
                    print(f"  {metric}: {score}/5")

            else:
                print(f"❌ API returned error: {result.get('error', 'Unknown error')}")

        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function."""
    print("AMR Prescription Evaluation API Test")
    print("=" * 50)
    print()

    # Get API URL
    api_url = os.getenv("API_BASE_URL")
    if not api_url:
        # Try to construct from common patterns
        api_url = "https://your-project.supabase.co/functions/v1/evaluate-prescription"
        print("⚠️  API_BASE_URL not set, using default template"        print("   Update API_BASE_URL environment variable or edit this script"
    print(f"📍 API URL: {api_url}")
    print()

    # Check if we have a Groq key (for reference, not used in this test)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ GROQ_API_KEY found (not needed for this test)")
    else:
        print("⚠️  GROQ_API_KEY not found (not needed for this test)")

    print()

    # Test the API
    test_evaluation_api(api_url, groq_key or "")

    print()
    print("=" * 50)
    print("Test complete!")
    print()
    print("Next steps:")
    print("1. If test failed, check Supabase function logs:")
    print("   supabase functions logs evaluate-prescription")
    print("2. Verify function is deployed:")
    print("   supabase functions list")
    print("3. Check environment variables in Supabase dashboard")

if __name__ == "__main__":
    main()
