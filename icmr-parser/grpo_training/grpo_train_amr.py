"""
GRPO Fine-tuning for AMR Prescription Validation.

This script implements Group Relative Policy Optimization (GRPO) for fine-tuning
language models on AMR prescription validation and general AMR query tasks.

Adapted from style transfer GRPO implementation for medical domain.
"""

import os
import json
import time
import asyncio
import aiohttp
import nest_asyncio
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datasets import Dataset, load_from_disk
from unsloth import FastLanguageModel
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')
nest_asyncio.apply()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./grpo_amr_model"

# Supabase Edge Function URL (UPDATE THIS after deployment)
API_BASE_URL = "https://your-project.supabase.co/functions/v1/evaluate-prescription"

# Priority metrics for training (faster, focused on key aspects)
PRIORITY_METRICS = ['clinical_accuracy', 'guideline_adherence', 'reasoning_completeness']

# All metrics for comprehensive evaluation
ALL_METRICS = [
    'clinical_accuracy',
    'guideline_adherence',
    'reasoning_completeness',
    'safety_awareness',
    'decision_appropriateness',
    'reference_accuracy'
]

# Reward weights (must sum to 1.0)
REWARD_WEIGHTS = {
    'clinical_accuracy': 0.25,
    'guideline_adherence': 0.25,
    'reasoning_completeness': 0.20,
    'safety_awareness': 0.15,
    'decision_appropriateness': 0.10,
    'reference_accuracy': 0.05
}

# API Configuration
API_CONFIG = {
    "max_concurrent_requests": 50,  # Reduced for medical evaluation
    "timeout_seconds": 45,  # Increased for complex medical reasoning
    "max_retries": 3,
    "retry_delay": 2.0,
    "use_cache": True,
}

# Evaluation frequency (evaluate every N training steps)
EVAL_FREQUENCY = 10  # Less frequent due to longer evaluation time

# GRPO Hyperparameters
GRPO_CONFIG = {
    "num_generations_per_prompt": 2,
    "batch_size": 2,  # Smaller batch for medical domain
    "learning_rate": 5e-6,  # Lower LR for medical fine-tuning
    "num_train_epochs": 3,
    "max_length": 1024,  # Longer for detailed medical reasoning
    "temperature": 0.7,
    "top_p": 0.95,
    "kl_coef": 0.05,
    "clip_range": 0.2,
    "vf_coef": 0.1,
    "eval_frequency": EVAL_FREQUENCY,
    "use_priority_metrics": True,  # Use priority metrics for faster training
}

# Unsloth Configuration
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

print("✅ Configuration loaded successfully!")
print(f"📍 API Base URL: {API_BASE_URL}")
print(f"🎯 Using Metrics: {'PRIORITY' if GRPO_CONFIG['use_priority_metrics'] else 'ALL'}")
print(f"📊 Evaluation Frequency: Every {EVAL_FREQUENCY} steps")


# ============================================================================
# API-INTEGRATED REWARD MODEL
# ============================================================================

class APIIntegratedRewardModel:
    """
    Reward model that integrates with Supabase Edge Function for evaluation.
    """
    
    def __init__(
        self,
        api_base_url: str,
        metrics: List[str],
        weights: Dict[str, float],
        api_config: Dict[str, Any],
        use_cache: bool = True
    ):
        self.api_base_url = api_base_url
        self.metrics = metrics
        self.weights = weights
        self.api_config = api_config
        self.use_cache = use_cache
        
        self.cache = {} if use_cache else None
        self.stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'api_errors': 0,
            'total_api_time': 0.0
        }
        
        print(f"✅ Reward Model initialized with {len(metrics)} metrics")
        print(f"   Metrics: {metrics}")
        print(f"   Caching: {'Enabled' if use_cache else 'Disabled'}")
    
    def _create_cache_key(self, context: Dict, model_output: str, ground_truth: str) -> str:
        """Create a unique cache key for an API call."""
        content = f"{json.dumps(context)}|{model_output}|{ground_truth}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _call_api_async(
        self,
        session: aiohttp.ClientSession,
        context: Dict,
        model_output: str,
        ground_truth: str,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Make async API call to evaluation endpoint."""
        
        # Check cache
        if self.use_cache:
            cache_key = self._create_cache_key(context, model_output, ground_truth)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        # Prepare request
        payload = {
            "patient_case": context,
            "model_output": model_output,
            "ground_truth": ground_truth,
            "metrics": self.metrics
        }
        
        try:
            start_time = time.time()
            
            async with session.post(
                self.api_base_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.api_config['timeout_seconds'])
            ) as response:
                elapsed = time.time() - start_time
                self.stats['total_api_time'] += elapsed
                self.stats['total_calls'] += 1
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Cache the result
                    if self.use_cache:
                        self.cache[cache_key] = result
                    
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"API returned status {response.status}: {error_text}")
        
        except Exception as e:
            # Retry logic
            if retry_count < self.api_config['max_retries']:
                await asyncio.sleep(self.api_config['retry_delay'] * (2 ** retry_count))
                return await self._call_api_async(
                    session, context, model_output, ground_truth, retry_count + 1
                )
            else:
                self.stats['api_errors'] += 1
                print(f"⚠️ API call failed after {retry_count} retries: {str(e)}")
                # Return default low scores
                return {
                    "success": False,
                    "evaluations": {metric: {"score": 1} for metric in self.metrics},
                    "weighted_reward": 0.0
                }
    
    async def _evaluate_batch_async(
        self,
        contexts: List[Dict],
        generated: List[str],
        references: List[str]
    ) -> List[Dict]:
        """Evaluate a batch of generations using async API calls."""
        semaphore = asyncio.Semaphore(self.api_config['max_concurrent_requests'])
        
        async def evaluate_single(ctx, gen, ref):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    return await self._call_api_async(session, ctx, gen, ref)
        
        tasks = [
            evaluate_single(ctx, gen, ref)
            for ctx, gen, ref in zip(contexts, generated, references)
        ]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def evaluate_batch(
        self,
        contexts: List[Dict],
        generated: List[str],
        references: List[str]
    ) -> List[Dict]:
        """Synchronous wrapper for batch evaluation."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._evaluate_batch_async(contexts, generated, references)
        )
    
    def compute_batch_rewards(
        self,
        contexts: List[Dict],
        generated: List[str],
        references: List[str]
    ) -> Tuple[List[float], List[Dict]]:
        """Compute rewards for a batch of generations."""
        results = self.evaluate_batch(contexts, generated, references)
        
        rewards = []
        all_scores = []
        
        for result in results:
            if result.get("success", False):
                reward = result.get("weighted_reward", 0.0)
                scores = result.get("evaluations", {})
            else:
                reward = 0.0
                scores = {}
            
            rewards.append(reward)
            all_scores.append(scores)
        
        return rewards, all_scores
    
    def print_stats(self):
        """Print cache and API usage statistics."""
        print(f"\n{'='*60}")
        print("Reward Model Statistics")
        print(f"{'='*60}")
        print(f"Total API calls: {self.stats['total_calls']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        if self.stats['total_calls'] > 0:
            cache_rate = (self.stats['cache_hits'] / (self.stats['total_calls'] + self.stats['cache_hits'])) * 100
            print(f"Cache hit rate: {cache_rate:.1f}%")
        print(f"API errors: {self.stats['api_errors']}")
        if self.stats['total_calls'] > 0:
            avg_time = self.stats['total_api_time'] / self.stats['total_calls']
            print(f"Average API call time: {avg_time:.2f}s")
        print(f"{'='*60}\n")


# ============================================================================
# GRPO TRAINER
# ============================================================================

class GRPOTrainer:
    """GRPO Trainer for AMR prescription validation."""
    
    def __init__(self, model, tokenizer, reward_model, config, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config
        self.output_dir = output_dir
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"]
        )
        
        self.metrics = {
            "epoch": [],
            "step": [],
            "loss": [],
            "mean_reward": [],
            "max_reward": [],
            "detailed_scores": [],
        }
        
        self.global_step = 0
        self.eval_frequency = config.get("eval_frequency", 10)
        
        print(f"✅ GRPO Trainer initialized")
        print(f"   Eval frequency: Every {self.eval_frequency} steps")
    
    def generate_responses(self, prompts: List[str], num_generations: int) -> List[List[str]]:
        """Generate multiple responses for each prompt."""
        self.model.eval()
        all_generations = []
        
        with torch.no_grad():
            for prompt in prompts:
                formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    num_return_sequences=num_generations,
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                generations = []
                for output in outputs:
                    decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                    if "<|assistant|>" in decoded:
                        decoded = decoded.split("<|assistant|>")[-1].strip()
                    generations.append(decoded)
                
                all_generations.append(generations)
        
        return all_generations
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages."""
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages.tolist()
    
    def compute_loss(self, prompts: List[str], generations: List[str], advantages: List[float]):
        """Compute the GRPO loss."""
        self.model.train()
        total_loss = 0.0
        
        for prompt, generation, advantage in zip(prompts, generations, advantages):
            formatted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{generation}"
            
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            weighted_loss = outputs.loss * advantage
            total_loss += weighted_loss
        
        return total_loss / len(prompts)
    
    def train_step(self, batch_prompts: List[str], batch_references: List[str], 
                   batch_contexts: List[Dict], evaluate_this_step: bool = False):
        """Single training step for GRPO."""
        self.global_step += 1
        
        # Generate responses
        all_generations = self.generate_responses(
            batch_prompts,
            self.config["num_generations_per_prompt"]
        )
        
        # Flatten
        flat_prompts = []
        flat_generations = []
        flat_references = []
        flat_contexts = []
        
        for prompt, generations, reference, context in zip(
            batch_prompts, all_generations, batch_references, batch_contexts
        ):
            for generation in generations:
                flat_prompts.append(prompt)
                flat_generations.append(generation)
                flat_references.append(reference)
                flat_contexts.append(context)
        
        # Compute rewards
        all_rewards = []
        all_scores = None
        
        if evaluate_this_step:
            all_rewards, all_scores = self.reward_model.compute_batch_rewards(
                flat_contexts, flat_generations, flat_references
            )
        else:
            # Simple heuristic rewards
            for gen, ref in zip(flat_generations, flat_references):
                len_ratio = len(gen.split()) / max(len(ref.split()), 1)
                reward = 1.0 - abs(1.0 - len_ratio)
                reward = max(0, min(1, reward)) * 0.5
                all_rewards.append(reward)
        
        # Compute advantages
        advantages = self.compute_advantages(all_rewards)
        
        # Update model
        self.optimizer.zero_grad()
        loss = self.compute_loss(flat_prompts, flat_generations, advantages)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "mean_reward": np.mean(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "detailed_scores": all_scores,
            "evaluated": evaluate_this_step
        }
    
    def train(self, dataset, num_epochs):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"🚀 Starting GRPO Training for AMR")
        print(f"{'='*80}")
        print(f"📊 Epochs: {num_epochs}")
        print(f"📦 Dataset size: {len(dataset)}")
        print(f"🔢 Batch size: {self.config['batch_size']}")
        print(f"⏱️  Eval frequency: Every {self.eval_frequency} steps")
        print(f"🎲 Generations per prompt: {self.config['num_generations_per_prompt']}")
        print(f"{'='*80}\n")
        
        epoch_pbar = tqdm(range(num_epochs), desc="📚 Epochs", position=0, leave=True)
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"📚 Epoch {epoch + 1}/{num_epochs}")
            
            epoch_metrics = {
                "loss": [],
                "mean_reward": [],
                "max_reward": [],
            }
            
            num_batches = (len(dataset) + self.config["batch_size"] - 1) // self.config["batch_size"]
            
            step_pbar = tqdm(
                range(0, len(dataset), self.config["batch_size"]),
                desc=f"  🔄 Steps",
                position=1,
                leave=False,
                total=num_batches
            )
            
            for i in step_pbar:
                batch = dataset[i:i + self.config["batch_size"]]
                batch_prompts = batch["prompt"]
                batch_references = batch["reference"]
                batch_contexts = batch["context"]
                
                evaluate_this_step = (self.global_step % self.eval_frequency == 0)
                
                metrics = self.train_step(
                    batch_prompts, batch_references, batch_contexts, evaluate_this_step
                )
                
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key].append(metrics[key])
                
                eval_marker = "📊 EVAL" if evaluate_this_step else "⚡ FAST"
                step_pbar.set_postfix({
                    'type': eval_marker,
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['mean_reward']:.4f}",
                    'max': f"{metrics['max_reward']:.4f}",
                    'global': self.global_step
                })
                
                if metrics.get('detailed_scores'):
                    self.metrics['detailed_scores'].append({
                        'step': self.global_step,
                        'scores': metrics['detailed_scores']
                    })
            
            step_pbar.close()
            
            # Epoch summary
            epoch_loss = np.mean(epoch_metrics["loss"])
            epoch_mean_reward = np.mean(epoch_metrics["mean_reward"])
            epoch_max_reward = np.mean(epoch_metrics["max_reward"])
            
            epoch_pbar.set_postfix({
                'loss': f"{epoch_loss:.4f}",
                'reward': f"{epoch_mean_reward:.4f}",
                'max': f"{epoch_max_reward:.4f}"
            })
            
            self.metrics["epoch"].append(epoch + 1)
            self.metrics["loss"].append(epoch_loss)
            self.metrics["mean_reward"].append(epoch_mean_reward)
            self.metrics["max_reward"].append(epoch_max_reward)
            
            # Save checkpoint
            checkpoint_dir = f"{self.output_dir}/checkpoint-epoch-{epoch + 1}"
            self.save_checkpoint(checkpoint_dir)
            
            print(f"\n📊 Epoch {epoch + 1} Summary: Loss={epoch_loss:.4f}, Reward={epoch_mean_reward:.4f}, Max={epoch_max_reward:.4f}")
        
        epoch_pbar.close()
        
        print(f"\n{'='*80}")
        print("✅ Training Complete!")
        print(f"{'='*80}\n")
        
        self.reward_model.print_stats()
    
    def save_checkpoint(self, checkpoint_dir):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        with open(f"{checkpoint_dir}/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    data_dir = Path(__file__).parent / "data"
    train_dataset_path = data_dir / "train_hf"
    
    print(f"\n📂 Loading dataset from: {train_dataset_path}")
    train_dataset = load_from_disk(str(train_dataset_path))
    print(f"✅ Loaded {len(train_dataset)} training examples")
    
    # Load model
    print(f"\n🔄 Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # Add LoRA
    print("\n🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("\n📊 Trainable Parameters:")
    model.print_trainable_parameters()
    
    # Initialize reward model
    metrics_to_use = PRIORITY_METRICS if GRPO_CONFIG["use_priority_metrics"] else ALL_METRICS
    
    reward_model = APIIntegratedRewardModel(
        api_base_url=API_BASE_URL,
        metrics=metrics_to_use,
        weights=REWARD_WEIGHTS,
        api_config=API_CONFIG,
        use_cache=API_CONFIG["use_cache"]
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=GRPO_CONFIG,
        output_dir=OUTPUT_DIR
    )
    
    # Start training
    trainer.train(
        dataset=train_dataset,
        num_epochs=GRPO_CONFIG["num_train_epochs"]
    )
    
    # Save final model
    final_model_dir = f"{OUTPUT_DIR}/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    print(f"\n✅ Final model saved to {final_model_dir}")


if __name__ == "__main__":
    main()

