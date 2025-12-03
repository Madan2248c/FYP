# AMR Prescription Evaluation Edge Function

This Supabase Edge Function serves as the LLM-as-a-Judge API for GRPO fine-tuning of the AMR prescription validation model.

## TypeScript Configuration

The function uses `@ts-ignore` comments to suppress TypeScript errors that occur in your IDE. These errors are normal because your IDE's TypeScript compiler doesn't understand Deno runtime types. The function will work perfectly when deployed to Supabase's Deno runtime environment.

**Why @ts-ignore?**
- Deno globals (`Deno.env`) are only available in Deno runtime
- Web APIs (`fetch`, `Response`) are available in Deno but not in Node.js TypeScript
- These imports work correctly in Supabase Edge Functions

The `deno.json` configuration file ensures proper compilation when deployed.

## Setup

### 1. Install Supabase CLI

```bash
# macOS
brew install supabase/tap/supabase

# Or using npm
npm install -g supabase
```

### 2. Initialize Supabase Project

```bash
cd /Users/madan.gopal/Desktop/clg/FYP/icmr-parser/grpo_training
supabase init
```

### 3. Set Environment Variables

Create a `.env` file in the function directory:

```bash
# In supabase/functions/evaluate-prescription/.env
GROQ_API_KEY=your_groq_api_key_here
```

Or set secrets in Supabase:

```bash
supabase secrets set GROQ_API_KEY=your_groq_api_key_here
```

### 4. Deploy the Function

```bash
# Deploy to Supabase
supabase functions deploy evaluate-prescription

# Or test locally first
supabase functions serve evaluate-prescription
```

## API Usage

### Endpoint

```
POST https://your-project.supabase.co/functions/v1/evaluate-prescription
```

### Request Format

```json
{
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
  "model_output": "Step 1: Patient assessment...\nStep 2: Diagnosis validation...\nApproved: Prescription is compliant.",
  "ground_truth": "Step 1: Patient Assessment...\nStep 2: Diagnosis Validation...\nApproved: Fully compliant.",
  "metrics": ["clinical_accuracy", "guideline_adherence", "reasoning_completeness"]
}
```

### Response Format

```json
{
  "success": true,
  "evaluations": {
    "clinical_accuracy": {
      "metric": "clinical_accuracy",
      "score": 5,
      "justification": "All medical facts and dosages are correct..."
    },
    "guideline_adherence": {
      "metric": "guideline_adherence",
      "score": 5,
      "justification": "Perfectly follows ICMR first-line recommendations..."
    },
    "reasoning_completeness": {
      "metric": "reasoning_completeness",
      "score": 4,
      "justification": "Covers most steps but could include more detail..."
    }
  },
  "weighted_reward": 0.925,
  "metrics_evaluated": 3,
  "model": "llama-3.3-70b-versatile"
}
```

## Evaluation Metrics

### Priority Metrics (Default)
- **clinical_accuracy** (25% weight) - Correctness of medical reasoning
- **guideline_adherence** (25% weight) - Alignment with ICMR 2025
- **reasoning_completeness** (20% weight) - Coverage of all reasoning steps

### Additional Metrics (Optional)
- **safety_awareness** (15% weight) - Identification of risks/contraindications
- **decision_appropriateness** (10% weight) - Correctness of Approve/Modify/Reject
- **reference_accuracy** (5% weight) - Accuracy of ICMR citations

## Testing Locally

```bash
# Start local function server
supabase functions serve evaluate-prescription

# Test with curl
curl -X POST http://localhost:54321/functions/v1/evaluate-prescription \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**TypeScript Configuration**: If you see TypeScript errors in your IDE about Deno imports or the `Deno` global, these are expected. The configuration files (`deno.json`, `tsconfig.json`) ensure proper type checking when deployed to Supabase's Deno runtime.

## Cost Optimization

- **Priority Metrics Mode**: Uses only 3 metrics (faster, cheaper)
- **All Metrics Mode**: Uses all 6 metrics (comprehensive but more API calls)
- **Caching**: Implement caching in your training code to avoid duplicate evaluations

## Integration with GRPO Training

```python
# In your GRPO training code
API_BASE_URL = "https://your-project.supabase.co/functions/v1/evaluate-prescription"

response = requests.post(
    API_BASE_URL,
    json={
        "patient_case": case,
        "model_output": generated_text,
        "ground_truth": reference_text,
        "metrics": ["clinical_accuracy", "guideline_adherence", "reasoning_completeness"]
    }
)

result = response.json()
reward = result["weighted_reward"]  # Use this for GRPO training
```

## Monitoring

Check function logs:

```bash
supabase functions logs evaluate-prescription
```

## Troubleshooting

1. **GROQ_API_KEY not found**: Make sure to set the secret in Supabase
2. **CORS errors**: The function includes CORS headers, but check your client configuration
3. **Timeout errors**: Groq API calls may take 2-5 seconds per metric. Consider using priority metrics only during training.

