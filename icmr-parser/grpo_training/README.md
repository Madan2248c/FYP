# GRPO Fine-tuning for AMR Prescription Validation

This directory contains the complete pipeline for fine-tuning a language model using **Group Relative Policy Optimization (GRPO)** on AMR (Antimicrobial Resistance) prescription validation tasks.

## 📋 Overview

The system consists of three main components:

1. **LLM-as-a-Judge Evaluation System** - Uses Groq Llama to score model outputs
2. **Supabase Edge Function** - Serverless API endpoint for evaluation
3. **GRPO Training Pipeline** - Fine-tunes Llama 3.1 8B on AMR tasks

## 🎯 Evaluation Metrics

The system evaluates model outputs on 6 key metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Clinical Accuracy** | 25% | Correctness of medical reasoning and facts |
| **Guideline Adherence** | 25% | Alignment with ICMR 2025 guidelines |
| **Reasoning Completeness** | 20% | Coverage of all required reasoning steps |
| **Safety Awareness** | 15% | Identification of risks/contraindications |
| **Decision Appropriateness** | 10% | Correctness of Approve/Modify/Reject |
| **Reference Accuracy** | 5% | Accuracy of ICMR citations |

### Priority Metrics Mode (Recommended for Training)
Uses only the top 3 metrics (Clinical Accuracy, Guideline Adherence, Reasoning Completeness) for faster training with 60% fewer API calls.

## 🚀 Quick Start

### Prerequisites

```bash
# Python packages
pip install unsloth transformers datasets groq pydantic tqdm aiohttp nest-asyncio

# Supabase CLI
brew install supabase/tap/supabase  # macOS
# or
npm install -g supabase
```

### Step 1: Prepare Dataset

```bash
cd /Users/madan.gopal/Desktop/clg/FYP/icmr-parser/grpo_training
python prepare_grpo_dataset.py
```

This converts your merged AMR dataset into GRPO format:
- Input: `../data/amr_merged_final_dataset.jsonl`
- Output: `data/train_grpo.json`, `data/val_grpo.json`

### Step 2: Deploy Evaluation API

```bash
# Initialize Supabase (if not already done)
supabase init

# Set your Groq API key
supabase secrets set GROQ_API_KEY=your_groq_api_key_here

# Deploy the edge function
supabase functions deploy evaluate-prescription

# Get your function URL
# It will be: https://YOUR_PROJECT_ID.supabase.co/functions/v1/evaluate-prescription
```

**Important:** Update `API_BASE_URL` in `grpo_train_amr.py` with your deployed function URL.

### Step 3: Run GRPO Training

```bash
# Make sure you have GPU access
python grpo_train_amr.py
```

Training configuration:
- **Model**: Llama 3.1 8B Instruct
- **Batch Size**: 2 (medical domain)
- **Learning Rate**: 5e-6
- **Epochs**: 3
- **Eval Frequency**: Every 10 steps
- **LoRA**: r=16, alpha=32

## 📁 Project Structure

```
grpo_training/
├── README.md                          # This file
├── evaluation_metrics.py              # Metric definitions
├── llm_judge.py                       # LLM-as-Judge implementation
├── prepare_grpo_dataset.py            # Dataset preparation
├── grpo_train_amr.py                  # Main training script
├── data/
│   ├── train_grpo.json               # Training data (GRPO format)
│   ├── val_grpo.json                 # Validation data
│   ├── train_hf/                     # HuggingFace dataset format
│   └── val_hf/
├── supabase/
│   └── functions/
│       └── evaluate-prescription/
│           ├── index.ts              # Edge function code
│           └── README.md             # Deployment guide
└── grpo_amr_model/                   # Output directory
    ├── checkpoint-epoch-1/
    ├── checkpoint-epoch-2/
    ├── checkpoint-epoch-3/
    └── final_model/
```

## 🔧 Configuration

### Training Configuration (`grpo_train_amr.py`)

```python
# Model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# API Endpoint (UPDATE THIS!)
API_BASE_URL = "https://your-project.supabase.co/functions/v1/evaluate-prescription"

# Metrics
PRIORITY_METRICS = ['clinical_accuracy', 'guideline_adherence', 'reasoning_completeness']

# GRPO Hyperparameters
GRPO_CONFIG = {
    "num_generations_per_prompt": 2,
    "batch_size": 2,
    "learning_rate": 5e-6,
    "num_train_epochs": 3,
    "max_length": 1024,
    "temperature": 0.7,
    "eval_frequency": 10,
}
```

### Evaluation API Configuration

```python
API_CONFIG = {
    "max_concurrent_requests": 50,
    "timeout_seconds": 45,
    "max_retries": 3,
    "retry_delay": 2.0,
    "use_cache": True,
}
```

## 📊 Dataset Format

### Input Format (GRPO)

```json
{
  "prompt": "**Role:** You are a clinical pharmacist...\n**Task:** Validate the following prescription...",
  "reference": "Step 1: Patient Assessment...\nStep 2: Diagnosis Validation...\nApproved: ...",
  "context": {
    "patient_profile": {...},
    "diagnosis": "Community Acquired Pneumonia",
    "prescription": {...}
  },
  "task_type": "prescription_validation"
}
```

### Evaluation API Request

```json
{
  "patient_case": {
    "patient_profile": {...},
    "diagnosis": "...",
    "prescription": {...}
  },
  "model_output": "Step 1: ...\nStep 2: ...\nApproved: ...",
  "ground_truth": "Step 1: ...\nStep 2: ...\nApproved: ...",
  "metrics": ["clinical_accuracy", "guideline_adherence", "reasoning_completeness"]
}
```

### Evaluation API Response

```json
{
  "success": true,
  "evaluations": {
    "clinical_accuracy": {
      "metric": "clinical_accuracy",
      "score": 5,
      "justification": "All medical facts are correct..."
    },
    "guideline_adherence": {
      "metric": "guideline_adherence",
      "score": 5,
      "justification": "Perfectly follows ICMR guidelines..."
    },
    "reasoning_completeness": {
      "metric": "reasoning_completeness",
      "score": 4,
      "justification": "Covers most steps..."
    }
  },
  "weighted_reward": 0.925,
  "metrics_evaluated": 3
}
```

## 💡 Training Strategy

### GRPO Algorithm

1. **Generate**: Create multiple outputs per prompt (2 generations)
2. **Evaluate**: Score each output using LLM-as-Judge
3. **Compute Advantages**: Normalize rewards within each group
4. **Update**: Train model to favor high-reward outputs

### Evaluation Frequency Optimization

- **Every Step**: Most accurate but expensive (100% API calls)
- **Every 5 Steps**: Good balance (20% API calls) 
- **Every 10 Steps**: Recommended for medical domain (10% API calls)
- **Every 20 Steps**: Fastest but less precise (5% API calls)

### Cost Optimization

```
Example: 100 training steps, batch_size=2, 2 generations per prompt

Priority Metrics (3 metrics):
- Eval every 10 steps: 10 × 4 × 3 = 120 API calls
- Eval every 5 steps: 20 × 4 × 3 = 240 API calls

All Metrics (6 metrics):
- Eval every 10 steps: 10 × 4 × 6 = 240 API calls
- Eval every 5 steps: 20 × 4 × 6 = 480 API calls

Savings: Priority + Eval-10 = 75% fewer API calls vs All + Eval-5
```

## 🧪 Testing

### Test LLM Judge Locally

```bash
# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Run test
python llm_judge.py
```

### Test Evaluation API

```bash
# Start local Supabase function
supabase functions serve evaluate-prescription

# Test with curl
curl -X POST http://localhost:54321/functions/v1/evaluate-prescription \
  -H "Content-Type: application/json" \
  -d '{
    "patient_case": {...},
    "model_output": "...",
    "ground_truth": "...",
    "metrics": ["clinical_accuracy"]
  }'
```

### Test Dataset Preparation

```bash
python prepare_grpo_dataset.py
```

Expected output:
```
✅ Loaded 201 examples
✅ Converted 201 examples
📊 Dataset Statistics:
  prescription_validation: 12 examples
  guideline_lookup: 50 examples
  drug_information: 30 examples
  ...
  Total: 201 examples
```

## 📈 Monitoring Training

### Progress Bars

```
📚 Epoch 1/3: 100%|████████| 3/3 [02:30<00:00, loss=2.1234, reward=0.7500]
  🔄 Steps: 100%|████████| 100/100 [02:30<00:00, type=📊 EVAL, loss=2.1234]
```

### Checkpoints

Models are saved after each epoch:
```
grpo_amr_model/
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
├── checkpoint-epoch-3/
└── final_model/
```

### Metrics JSON

Training metrics are saved in each checkpoint:
```json
{
  "epoch": [1, 2, 3],
  "loss": [2.5, 2.1, 1.8],
  "mean_reward": [0.65, 0.75, 0.82],
  "max_reward": [0.85, 0.90, 0.95],
  "detailed_scores": [...]
}
```

## 🎯 Expected Results

### Baseline (Before Training)
- Clinical Accuracy: ~2.5/5
- Guideline Adherence: ~2.0/5
- Reasoning Completeness: ~2.5/5
- Weighted Reward: ~0.40

### After 3 Epochs
- Clinical Accuracy: ~4.5/5
- Guideline Adherence: ~4.5/5
- Reasoning Completeness: ~4.0/5
- Weighted Reward: ~0.85

## 🚨 Troubleshooting

### API Connection Issues

```python
# Test API connectivity
import requests
response = requests.post(
    "https://your-project.supabase.co/functions/v1/evaluate-prescription",
    json={"patient_case": {}, "model_output": "test", "ground_truth": "test"}
)
print(response.status_code, response.json())
```

### CUDA Out of Memory

```python
# Reduce batch size
GRPO_CONFIG["batch_size"] = 1

# Or reduce max_length
GRPO_CONFIG["max_length"] = 512
```

### Slow Training

```python
# Increase eval frequency
GRPO_CONFIG["eval_frequency"] = 20

# Use priority metrics only
GRPO_CONFIG["use_priority_metrics"] = True
```

## 📚 References

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **Unsloth**: [Fast LLM Fine-tuning](https://github.com/unslothai/unsloth)
- **ICMR Guidelines**: [2025 Antimicrobial Treatment Guidelines](https://icmr.nic.in/)

## 🤝 Contributing

1. Test changes with small dataset first
2. Monitor API costs during development
3. Use priority metrics for faster iteration
4. Document any configuration changes

## 📝 License

This project is part of the ICMR Parser system for AMR research and education.

---

**Next Steps:**
1. ✅ Deploy Supabase Edge Function
2. ✅ Prepare training dataset
3. ✅ Run GRPO training
4. 📊 Evaluate on validation set
5. 🚀 Deploy fine-tuned model

