# ICMR Parser Project - Complete Workflow Documentation

## 🎯 Project Overview

The ICMR Parser project extracts structured medical data from ICMR antimicrobial guidelines (PDF format) and creates comprehensive training datasets for building an AMR (Antimicrobial Resistance) LLM. The system generates two types of datasets:

1. **Prescription Validation Dataset**: Patient cases with correct/incorrect prescriptions for training reasoning capabilities
2. **General AMR Q&A Dataset**: Question-answer pairs covering guideline lookups, drug information, and treatment recommendations

---

## 📁 Project Structure

```
/Users/madan.gopal/Desktop/clg/FYP/icmr-parser/
├── main.py                           # Main entry point for the extraction pipeline
├── AMR_LLM_STRATEGY.md               # High-level strategy document
├── config/
│   └── config.yaml                   # Configuration settings
├── data/
│   ├── raw/                         # Original PDF source
│   ├── extracted/                   # Raw text extracted from PDF
│   ├── structured/                  # Final structured JSON data
│   └── [various dataset files]      # Training datasets
└── src/
    ├── agents/                      # Multi-agent system components
    ├── prompts/                     # LLM prompt templates
    ├── utils.py                     # Utility functions
    ├── data_models.py               # Pydantic data models
    ├── pdf_parser.py                # PDF text extraction
    ├── llm_extractor.py             # LLM-based structured extraction
    ├── generate_patient_cases.py    # Synthetic patient case generation
    ├── generate_amr_training_data.py # Q&A dataset generation
    ├── multi_agent_refiner.py       # Prescription validation reasoning
    └── merge_datasets.py            # Dataset unification
```

---

## 🔄 Complete Workflow Steps

### Phase 1: PDF Processing & Structured Data Extraction

#### Step 1: PDF Parsing (`main.py` → `pdf_parser.py`)
**Purpose**: Extract raw text and identify syndrome sections from the ICMR PDF.

**Key Files:**
- `main.py`: Entry point that orchestrates the entire pipeline
- `src/pdf_parser.py`: Core PDF parsing logic using PyMuPDF

**Process:**
1. Load PDF: `NTG-Version-31st-July-final.pdf`
2. Extract text and tables from each page
3. Identify syndrome sections using keyword detection
4. Save extracted content to `data/extracted/` directory

**Command:**
```bash
python main.py --pdf-path data/raw/NTG-Version-31st-July-final.pdf
```

#### Step 2: LLM-Based Structured Extraction (`llm_extractor.py`)
**Purpose**: Convert raw text into structured JSON using Groq API with Llama models.

**Key Components:**
- `src/llm_extractor.py`: Handles LLM calls with automatic API key rotation
- `src/data_models.py`: Pydantic models defining the structured data format
- Uses LangChain's structured output for reliable JSON extraction

**Process:**
1. Load extracted text sections
2. For each syndrome section, call LLM with structured prompts
3. Extract: syndrome name, drugs, pathogens, special populations, contraindications
4. Save structured data to `data/structured/` directory
5. Generate validation report

**Output Format:**
```json
{
  "syndrome_name": "Community Acquired Pneumonia",
  "first_line_drugs": [
    {
      "drug_name": "PIPERACILLIN-TAZOBACTAM",
      "dosage": "4.5g",
      "route": "IV",
      "duration": "7 days"
    }
  ],
  "common_pathogens": [
    {
      "organism_name": "Streptococcus pneumoniae",
      "prevalence": "most common"
    }
  ]
}
```

---

### Phase 2: Dataset Generation for AMR LLM Training

#### Step 3: Generate General Q&A Dataset (`generate_amr_training_data.py`)
**Purpose**: Create question-answer pairs for general AMR knowledge.

**Types of Q&A Generated:**
1. **Guideline Lookups** (50 examples): "What is the first-line treatment for pneumonia?"
2. **Drug Information** (30 examples): "In which infections is Cefuroxime used?"
3. **Pathogen Treatment** (30 examples): "What antibiotics treat MRSA?"
4. **Educational Content** (5 examples): General AMR principles
5. **Syndrome Overviews** (30 examples): Comprehensive syndrome information

**Process:**
1. Load structured ICMR data from `data/structured/all_syndromes.json`
2. Use templates to generate diverse questions
3. Generate formatted responses with citations
4. Save to `data/amr_training_data.jsonl`

**Example Output:**
```json
{
  "task_type": "guideline_lookup",
  "query": "What is the first-line treatment for Community Acquired Pneumonia?",
  "response": "**First-Line Treatment:**\n- PIPERACILLIN-TAZOBACTAM 4.5g (IV) for 7 days\n- AZITHROMYCIN 500mg (PO) for 7 days\n\n**Common Pathogens:** Streptococcus pneumoniae, Haemophilus influenzae\n\n*Reference: ICMR 2025 Guidelines*"
}
```

#### Step 4: Generate Synthetic Patient Cases (`generate_patient_cases.py`)
**Purpose**: Create realistic patient cases with correct and incorrect prescriptions.

**Process:**
1. Load structured ICMR data
2. Generate patient profiles (age, gender, comorbidities)
3. Create symptoms based on syndrome
4. Generate prescriptions (60% correct, 40% incorrect)
5. Save to `data/patient_cases_*.json`

**Example Case:**
```json
{
  "case_id": "CAP_001",
  "patient_profile": {
    "age": 45,
    "gender": "male",
    "comorbidities": ["diabetes"],
    "allergies": []
  },
  "diagnosis": "Community Acquired Pneumonia",
  "symptoms": ["fever", "cough", "shortness of breath"],
  "prescription": {
    "drug": "PIPERACILLIN-TAZOBACTAM",
    "dosage": "4.5g IV",
    "duration": "7 days"
  },
  "is_correct": true
}
```

#### Step 5: Multi-Agent Reasoning Refinement (`multi_agent_refiner.py`)
**Purpose**: Generate step-by-step reasoning for prescription validation using Generator and Verifier agents.

**Key Components:**
- `src/agents/generator_agent.py`: Creates initial reasoning traces
- `src/agents/verifier_agent.py`: Validates reasoning accuracy
- `src/multi_agent_refiner.py`: Coordinates the agents with retry logic

**Process:**
1. Load patient cases from previous step
2. For each case, Generator agent creates reasoning steps
3. Verifier agent checks accuracy against ICMR guidelines
4. If incorrect, regenerate (max 2 retries)
5. Save validated reasoning to `data/reasoning_*/reasoning_dataset_validated.jsonl`

**Example Output:**
```json
{
  "instruction": "Validate prescription according to ICMR 2025 guidelines...",
  "context": {
    "patient_profile": {"age": 45, "comorbidities": ["diabetes"]},
    "diagnosis": "Community Acquired Pneumonia",
    "prescription": {"drug": "PIPERACILLIN-TAZOBACTAM", "dosage": "4.5g IV"}
  },
  "generator_reasoning_steps": [
    "Step 1: Patient has CAP with diabetes comorbidity",
    "Step 2: ICMR recommends Piperacillin-Tazobactam 4.5g IV for 7 days as first-line",
    "Step 3: Dosage matches guidelines exactly",
    "Step 4: No contraindications for diabetic patient"
  ],
  "final_answer": "Approved: The prescription for Piperacillin-Tazobactam 4.5g IV for 7 days is fully compliant with ICMR 2025 guidelines for Community Acquired Pneumonia",
  "icmr_reference": "Page 108, ICMR 2025 Guidelines"
}
```

---

### Phase 3: Dataset Unification & Training Preparation

#### Step 6: Merge Datasets (`merge_datasets.py`)
**Purpose**: Combine Q&A and prescription validation datasets into unified format.

**Process:**
1. Load prescription validation data from `data/reasoning_*/reasoning_dataset_validated.jsonl`
2. Load Q&A data from `data/amr_training_data.jsonl`
3. Convert to unified format with consistent structure
4. Balance task types (40% validation, 60% Q&A)
5. Save to `data/amr_merged_final_dataset.jsonl`

**Unified Format:**
```json
{
  "task_type": "prescription_validation", // or "general_query"
  "input": "...",  // Query or full case context
  "output": "...", // Answer or validation result
  "reasoning": [...], // Step-by-step reasoning (for validation)
  "reference": "...", // ICMR citation
  "metadata": {...}  // Additional context
}
```

---

## 🛠️ Key Components Deep Dive

### Configuration System (`config/config.yaml`)
- LLM settings (model, temperature, retries)
- PDF parsing options (table detection, OCR)
- Syndrome detection keywords
- Output preferences

### Utility Functions (`src/utils.py`)
- Logging setup with timestamps
- Configuration loading
- PDF validation
- Cost estimation for LLM calls
- Validation report generation

### Data Models (`src/data_models.py`)
Pydantic models ensuring data consistency:
- `DrugRecommendation`: Complete drug information
- `Pathogen`: Microorganism details
- `SpecialPopulation`: Population-specific adjustments
- `Contraindication`: Drug restrictions
- `SyndromeData`: Main structured format

### Agent System (`src/agents/`)
- **Generator Agent**: Creates reasoning traces for prescriptions
- **Verifier Agent**: Validates reasoning accuracy with dosage tolerance
- **Coordinator**: Manages agent interaction with retry logic

---

## 📊 Data Flow & Dependencies

```
PDF Source
    ↓
PDF Parser → Raw Text (data/extracted/)
    ↓
LLM Extractor → Structured JSON (data/structured/)
    ↓
├── Q&A Generator → General Queries (amr_training_data.jsonl)
│
└── Patient Case Generator → Synthetic Cases (patient_cases.json)
    ↓
    Multi-Agent Refiner → Reasoning Traces (reasoning_dataset_validated.jsonl)
    ↓
    Dataset Merger → Unified Dataset (amr_merged_final_dataset.jsonl)
    ↓
    Fine-tuning → AMR LLM
```

---

## 🚀 Usage Commands

### Complete Pipeline (End-to-End)
```bash
# 1. Extract structured data from PDF
python main.py --pdf-path data/raw/NTG-Version-31st-July-final.pdf

# 2. Generate Q&A dataset
python src/generate_amr_training_data.py

# 3. Generate patient cases
python src/generate_patient_cases.py --output data/patient_cases_large.json --count 200

# 4. Generate reasoning traces
python src/multi_agent_refiner.py \
  --input data/patient_cases_200_all.json \
  --output data/reasoning_200/ \
  --generator-model gemini-2.0-flash-exp

# 5. Merge datasets
python src/merge_datasets.py
```

### Individual Components
```bash
# Skip PDF parsing if already done
python main.py --pdf-path data/raw/NTG-Version-31st-July-final.pdf --skip-parsing

# Generate more patient cases
python src/generate_patient_cases.py --count 500 --correct-ratio 0.7

# Resume interrupted reasoning generation
python src/multi_agent_refiner.py --resume
```

---

## 📈 Dataset Statistics

**Current Datasets:**
- **Structured Syndromes**: 67+ syndromes extracted from ICMR PDF
- **Q&A Examples**: 145+ general AMR questions and answers
- **Prescription Cases**: 56+ validated reasoning traces
- **Patient Cases**: 200+ synthetic cases available

**Target Goals:**
- 100-150 prescription validation cases
- 500+ Q&A examples
- 600-700 total training examples
- Unified dataset for multi-task fine-tuning

---

## 🎯 Final Output: AMR LLM Capabilities

The trained LLM can handle:

1. **General Queries**: "What is first-line treatment for pneumonia?"
2. **Prescription Validation**: Full patient case + prescription → approval/rejection with reasoning
3. **Drug Information**: "What is the dose of Vancomycin for MRSA?"
4. **Educational Content**: Antimicrobial stewardship guidance
5. **Safety Checks**: Allergy detection, contraindications, renal dosing

---

## 🔧 Key Technologies

- **PDF Processing**: PyMuPDF (fitz)
- **LLM API**: Groq (Llama 3.3 70B), Google Gemini
- **Data Validation**: Pydantic models
- **Parallel Processing**: ThreadPoolExecutor for API calls
- **Structured Output**: LangChain's JSON mode
- **Quality Assurance**: Multi-agent verification system

---

## 📚 File Purpose Summary

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestration and CLI interface |
| `src/pdf_parser.py` | Extract text/tables from PDF |
| `src/llm_extractor.py` | Convert text to structured JSON |
| `src/generate_amr_training_data.py` | Create Q&A training examples |
| `src/generate_patient_cases.py` | Generate synthetic patient cases |
| `src/multi_agent_refiner.py` | Create reasoning traces for validation |
| `src/merge_datasets.py` | Unify different dataset types |
| `src/data_models.py` | Define data structures |
| `src/utils.py` | Common utility functions |
| `config/config.yaml` | Configuration settings |

This workflow transforms raw PDF guidelines into comprehensive training data for building a medical AI system that can both answer questions and validate prescriptions according to ICMR standards.
