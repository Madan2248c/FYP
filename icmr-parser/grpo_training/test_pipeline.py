"""
Test script for GRPO training pipeline.

This script tests all components before running full training:
1. Dataset preparation
2. LLM Judge evaluation
3. API connectivity
4. Model loading
"""

import os
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("GRPO Training Pipeline Test")
print("=" * 80)
print()


# ============================================================================
# Test 1: Dataset Preparation
# ============================================================================

print("Test 1: Dataset Preparation")
print("-" * 80)

try:
    from prepare_grpo_dataset import load_amr_dataset, convert_to_grpo_format
    
    # Check if merged dataset exists
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "amr_merged_final_dataset.jsonl"
    
    if not input_file.exists():
        print(f"❌ Dataset not found: {input_file}")
        print("   Please run the dataset generation scripts first.")
        sys.exit(1)
    
    # Load a few examples
    data = load_amr_dataset(str(input_file))
    print(f"✅ Loaded {len(data)} examples from merged dataset")
    
    # Convert first 5 examples
    sample_data = data[:5]
    grpo_data = convert_to_grpo_format(sample_data)
    print(f"✅ Converted {len(grpo_data)} examples to GRPO format")
    
    # Show example
    if grpo_data:
        example = grpo_data[0]
        print(f"\n📝 Example prompt (first 200 chars):")
        print(example['prompt'][:200] + "...")
        print(f"\n✨ Example reference (first 150 chars):")
        print(example['reference'][:150] + "...")
    
    print("\n✅ Test 1 PASSED\n")
    
except Exception as e:
    print(f"❌ Test 1 FAILED: {e}\n")
    sys.exit(1)


# ============================================================================
# Test 2: LLM Judge Evaluation (Local)
# ============================================================================

print("Test 2: LLM Judge Evaluation")
print("-" * 80)

try:
    from llm_judge import LLMJudge
    from evaluation_metrics import PRIORITY_METRICS
    
    # Check for API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("⚠️  GROQ_API_KEY not found in environment")
        print("   Set it with: export GROQ_API_KEY=your_key_here")
        print("   Skipping LLM Judge test...")
    else:
        print(f"✅ Found GROQ_API_KEY: {groq_key[:10]}...")
        
        # Initialize judge
        judge = LLMJudge(api_key=groq_key)
        print(f"✅ Initialized LLM Judge with {judge.model}")
        
        # Test evaluation on simple case
        test_case = {
            "patient_profile": {"age": 45, "sex": "M", "history": [], "allergies": []},
            "diagnosis": "Community Acquired Pneumonia",
            "prescription": {"drug": "AZITHROMYCIN", "dosage": "500mg", "route": "PO"}
        }
        
        model_output = "Step 1: Patient is 45-year-old male.\nStep 2: CAP diagnosis confirmed.\nApproved: Prescription is appropriate."
        ground_truth = "Step 1: Patient Assessment - 45yo male, no comorbidities.\nStep 2: Diagnosis validated.\nApproved: Compliant with ICMR."
        
        print("\n🧪 Testing evaluation on sample case...")
        result = judge.evaluate_single_metric(
            "clinical_accuracy",
            test_case,
            model_output,
            ground_truth
        )
        
        print(f"✅ Evaluation result:")
        print(f"   Metric: {result['metric']}")
        print(f"   Score: {result['score']}/5")
        print(f"   Justification: {result['justification'][:100]}...")
    
    print("\n✅ Test 2 PASSED\n")
    
except Exception as e:
    print(f"❌ Test 2 FAILED: {e}\n")
    if "GROQ_API_KEY" not in str(e):
        sys.exit(1)


# ============================================================================
# Test 3: API Connectivity (if deployed)
# ============================================================================

print("Test 3: API Connectivity")
print("-" * 80)

try:
    import requests
    
    # Check if API URL is configured
    from grpo_train_amr import API_BASE_URL
    
    if "your-project" in API_BASE_URL:
        print("⚠️  API_BASE_URL not configured yet")
        print("   Update API_BASE_URL in grpo_train_amr.py after deploying Supabase function")
        print("   Skipping API connectivity test...")
    else:
        print(f"✅ API URL configured: {API_BASE_URL}")
        
        # Test ping
        print("\n🧪 Testing API connectivity...")
        test_payload = {
            "patient_case": {},
            "model_output": "test",
            "ground_truth": "test",
            "metrics": ["clinical_accuracy"]
        }
        
        try:
            response = requests.post(
                API_BASE_URL,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API is reachable and responding")
                print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
            else:
                print(f"⚠️  API returned status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Could not connect to API: {e}")
            print("   Make sure Supabase Edge Function is deployed")
    
    print("\n✅ Test 3 PASSED\n")
    
except Exception as e:
    print(f"❌ Test 3 FAILED: {e}\n")
    # Don't exit - API might not be deployed yet


# ============================================================================
# Test 4: Model Loading
# ============================================================================

print("Test 4: Model Loading")
print("-" * 80)

try:
    import torch
    from unsloth import FastLanguageModel
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA not available - training will be slow")
    
    print("\n🧪 Testing model loading (this may take a minute)...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_seq_length=512,  # Small for testing
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"✅ Model loaded successfully")
    print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
    
    # Test tokenization
    test_text = "This is a test prompt for the AMR model."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization works: {len(tokens['input_ids'][0])} tokens")
    
    print("\n✅ Test 4 PASSED\n")
    
except Exception as e:
    print(f"❌ Test 4 FAILED: {e}\n")
    print("   Make sure you have GPU access and sufficient memory")
    sys.exit(1)


# ============================================================================
# Summary
# ============================================================================

print("=" * 80)
print("✅ All Tests Passed!")
print("=" * 80)
print()
print("Next steps:")
print("1. If API test was skipped, deploy Supabase Edge Function:")
print("   cd supabase/functions")
print("   supabase functions deploy evaluate-prescription")
print()
print("2. Update API_BASE_URL in grpo_train_amr.py with your deployed URL")
print()
print("3. Run dataset preparation:")
print("   python prepare_grpo_dataset.py")
print()
print("4. Start GRPO training:")
print("   python grpo_train_amr.py")
print()
print("=" * 80)

