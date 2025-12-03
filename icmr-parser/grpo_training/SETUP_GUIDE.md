# Complete Setup Guide for GRPO Fine-tuning

This guide walks you through the complete setup process for GRPO fine-tuning of your AMR prescription validation model.

## 📋 Prerequisites

### 1. Hardware Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (A100, A6000, or similar)
  - For 40GB: Can train with batch_size=4
  - For 24GB: Use batch_size=2
  - For 16GB: Use batch_size=1 + gradient_checkpointing
- **RAM**: At least 32GB system RAM
- **Storage**: 50GB free space for model and checkpoints

### 2. Software Requirements

- **Python**: 3.10 or 3.11 (3.12 may have compatibility issues)
- **CUDA**: 11.8 or 12.1
- **Node.js**: 18+ (for Supabase CLI)

### 3. API Keys

- **Groq API Key**: Get from [console.groq.com](https://console.groq.com)
- **Supabase Account**: Sign up at [supabase.com](https://supabase.com)

## 🚀 Step-by-Step Setup

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd /Users/madan.gopal/Desktop/clg/FYP/icmr-parser/grpo_training

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Expected time**: 10-15 minutes

### Step 2: Set Environment Variables

```bash
# Create .env file
cat > .env << EOF
GROQ_API_KEY=your_groq_api_key_here
GROQ_API_KEY_1=your_first_key
GROQ_API_KEY_2=your_second_key  # Optional: for rate limit rotation
EOF

# Load environment variables
source .env  # On macOS/Linux
# or
set -a; source .env; set +a  # Alternative
```

**Tip**: Get multiple Groq API keys for higher rate limits during training.

### Step 3: Verify Dataset

```bash
# Check if merged dataset exists
ls -lh ../data/amr_merged_final_dataset.jsonl

# Should show something like:
# -rw-r--r--  1 user  staff   2.5M Dec  3 10:00 amr_merged_final_dataset.jsonl
```

If the file doesn't exist, run the dataset generation pipeline first:

```bash
cd /Users/madan.gopal/Desktop/clg/FYP/icmr-parser
python src/merge_datasets.py
```

### Step 4: Install Supabase CLI

```bash
# macOS
brew install supabase/tap/supabase

# Or using npm
npm install -g supabase

# Verify installation
supabase --version
```

### Step 5: Deploy Evaluation API

#### Option A: Automated Deployment (Recommended)

```bash
# Run deployment script
./deploy_supabase.sh
```

Follow the prompts to:
1. Enter your Groq API key
2. Initialize Supabase project
3. Deploy the edge function
4. Get your function URL

#### Option B: Manual Deployment

```bash
# Initialize Supabase
supabase init

# Login to Supabase
supabase login

# Link to your project (or create new one)
supabase link --project-ref your-project-ref

# Set secrets
supabase secrets set GROQ_API_KEY=your_groq_api_key_here

# Deploy function
supabase functions deploy evaluate-prescription

# Get function URL
supabase projects list
# URL will be: https://YOUR_PROJECT_ID.supabase.co/functions/v1/evaluate-prescription
```

**Expected time**: 5-10 minutes

### Step 6: Update Configuration

Edit `grpo_train_amr.py` and update the API URL:

```python
# Line ~15
API_BASE_URL = "https://YOUR_PROJECT_ID.supabase.co/functions/v1/evaluate-prescription"
```

### Step 7: Run Tests

```bash
# Test all components
python test_pipeline.py
```

Expected output:
```
Test 1: Dataset Preparation
✅ Loaded 201 examples from merged dataset
✅ Converted 201 examples to GRPO format
✅ Test 1 PASSED

Test 2: LLM Judge Evaluation
✅ Found GROQ_API_KEY: gsk_...
✅ Initialized LLM Judge with llama-3.3-70b-versatile
✅ Evaluation result:
   Metric: clinical_accuracy
   Score: 4/5
✅ Test 2 PASSED

Test 3: API Connectivity
✅ API URL configured: https://...
✅ API is reachable and responding
✅ Test 3 PASSED

Test 4: Model Loading
✅ PyTorch version: 2.8.0
✅ CUDA available: True
✅ CUDA device: NVIDIA A100-SXM4-40GB
✅ Model loaded successfully
✅ Test 4 PASSED

✅ All Tests Passed!
```

If any test fails, see [Troubleshooting](#troubleshooting) section.

### Step 8: Prepare Training Dataset

```bash
# Convert merged dataset to GRPO format
python prepare_grpo_dataset.py
```

Expected output:
```
📂 Loading dataset from: ../data/amr_merged_final_dataset.jsonl
✅ Loaded 201 examples
🔄 Converting to GRPO format...
✅ Converted 201 examples

📊 Dataset Statistics:
  prescription_validation: 12 examples
  guideline_lookup: 50 examples
  drug_information: 30 examples
  pathogen_treatment: 30 examples
  educational_content: 5 examples
  syndrome_overview: 30 examples
  general_query: 44 examples
  Total: 201 examples

📊 Split: 190 train, 11 validation
💾 Saving datasets...
✅ Saved to data/train_grpo.json
✅ Saved to data/val_grpo.json
✅ Dataset preparation complete!
```

**Expected time**: 1-2 minutes

### Step 9: Start Training

```bash
# Start GRPO training
python grpo_train_amr.py
```

**Expected time**: 
- Small dataset (200 examples): 2-3 hours
- Medium dataset (500 examples): 5-6 hours
- Large dataset (1000+ examples): 10-12 hours

Training output:
```
🚀 Starting GRPO Training for AMR
================================================================================
📊 Epochs: 3
📦 Dataset size: 190
🔢 Batch size: 2
⏱️  Eval frequency: Every 10 steps
🎲 Generations per prompt: 2

📚 Epoch 1/3: 100%|████████| loss=2.1234, reward=0.7500
  🔄 Steps: 100%|████████| type=📊 EVAL, loss=2.1234, reward=0.7500

📊 Epoch 1 Summary: Loss=2.1234, Reward=0.7500, Max=0.8500
💾 Checkpoint saved: grpo_amr_model/checkpoint-epoch-1

...

✅ Training Complete!
✅ Final model saved to grpo_amr_model/final_model
```

## 📊 Monitoring Training

### Real-time Monitoring

```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# Monitor API function logs
supabase functions logs evaluate-prescription --follow
```

### Check Progress

```bash
# View saved checkpoints
ls -lh grpo_amr_model/

# View metrics
cat grpo_amr_model/checkpoint-epoch-1/metrics.json | python -m json.tool
```

## 🎯 Post-Training

### Test the Trained Model

```python
from unsloth import FastLanguageModel

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./grpo_amr_model/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Test on a case
prompt = """**Role:** You are a clinical pharmacist...
**Task:** Validate the following prescription...

**Patient Information:**
- Age: 45 years
- Sex: M
- Diagnosis: Community Acquired Pneumonia
- Prescription: AZITHROMYCIN 500mg PO for 7 days
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Push to Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repository
repo_id = "your-username/amr-prescription-validator"
api.create_repo(repo_id=repo_id, private=False, exist_ok=True)

# Upload model
api.upload_folder(
    folder_path="./grpo_amr_model/final_model",
    repo_id=repo_id,
    path_in_repo="",
)

print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
```

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# In grpo_train_amr.py, reduce batch size
GRPO_CONFIG["batch_size"] = 1

# Or reduce max_length
GRPO_CONFIG["max_length"] = 512

# Or use gradient accumulation
GRPO_CONFIG["gradient_accumulation_steps"] = 4
```

### Issue 2: API Connection Timeout

**Error**: `aiohttp.ClientTimeout`

**Solutions**:
```python
# In grpo_train_amr.py, increase timeout
API_CONFIG["timeout_seconds"] = 60

# Or reduce concurrent requests
API_CONFIG["max_concurrent_requests"] = 25
```

### Issue 3: Groq Rate Limits

**Error**: `Rate limit exceeded`

**Solutions**:
1. Add multiple API keys in `.env`:
   ```bash
   GROQ_API_KEY_1=key1
   GROQ_API_KEY_2=key2
   GROQ_API_KEY_3=key3
   ```

2. Increase eval frequency:
   ```python
   GRPO_CONFIG["eval_frequency"] = 20  # Evaluate less often
   ```

3. Use priority metrics only:
   ```python
   GRPO_CONFIG["use_priority_metrics"] = True
   ```

### Issue 4: Supabase Function Not Responding

**Check function logs**:
```bash
supabase functions logs evaluate-prescription
```

**Common issues**:
- GROQ_API_KEY not set: `supabase secrets set GROQ_API_KEY=...`
- Function not deployed: `supabase functions deploy evaluate-prescription`
- Wrong URL: Check project ID in Supabase dashboard

### Issue 5: Dataset Not Found

**Error**: `FileNotFoundError: amr_merged_final_dataset.jsonl`

**Solution**:
```bash
# Go back to main project directory
cd /Users/madan.gopal/Desktop/clg/FYP/icmr-parser

# Run dataset generation
python src/generate_amr_training_data.py
python src/multi_agent_refiner.py
python src/merge_datasets.py

# Verify file exists
ls -lh data/amr_merged_final_dataset.jsonl
```

## 📚 Configuration Reference

### Training Hyperparameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `batch_size` | 2 | Training batch size | Reduce if OOM |
| `learning_rate` | 5e-6 | Learning rate | Lower for stability |
| `num_train_epochs` | 3 | Number of epochs | More for larger datasets |
| `max_length` | 1024 | Max sequence length | Reduce for speed |
| `temperature` | 0.7 | Generation temperature | Lower for consistency |
| `eval_frequency` | 10 | Steps between evals | Higher for speed |

### API Configuration

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `max_concurrent_requests` | 50 | Parallel API calls | Reduce if rate limited |
| `timeout_seconds` | 45 | Request timeout | Increase if timing out |
| `max_retries` | 3 | Retry attempts | Increase for reliability |
| `use_cache` | True | Cache responses | Keep enabled |

### Evaluation Metrics

| Metric | Weight | Priority | Description |
|--------|--------|----------|-------------|
| Clinical Accuracy | 25% | ✅ Yes | Medical facts correctness |
| Guideline Adherence | 25% | ✅ Yes | ICMR compliance |
| Reasoning Completeness | 20% | ✅ Yes | All steps covered |
| Safety Awareness | 15% | No | Risk identification |
| Decision Appropriateness | 10% | No | Correct decision |
| Reference Accuracy | 5% | No | Citation accuracy |

## 🎓 Best Practices

1. **Start Small**: Test with 50-100 examples first
2. **Monitor Costs**: Track Groq API usage
3. **Save Checkpoints**: Keep all epoch checkpoints
4. **Use Priority Metrics**: During training for speed
5. **Comprehensive Eval**: Use all metrics for final evaluation
6. **Version Control**: Track hyperparameters and results
7. **Test Regularly**: Validate on held-out test set

## 📞 Support

If you encounter issues:

1. Check logs: `supabase functions logs evaluate-prescription`
2. Test components: `python test_pipeline.py`
3. Review configuration in `grpo_train_amr.py`
4. Check GPU memory: `nvidia-smi`
5. Verify API connectivity with curl

## 🎉 Success Checklist

- [ ] Environment setup complete
- [ ] Groq API key configured
- [ ] Supabase Edge Function deployed
- [ ] API connectivity verified
- [ ] Dataset prepared (GRPO format)
- [ ] Test pipeline passes all tests
- [ ] Training started successfully
- [ ] Checkpoints saving correctly
- [ ] Final model saved
- [ ] Model tested on validation set

---

**Ready to train?** Run `python grpo_train_amr.py` and let it run! 🚀

