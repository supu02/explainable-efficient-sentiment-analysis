"""TinyBERT sentiment model with dynamic INT8 linear layers and CPU inference.

Uses the pretrained four-layer TinyBERT architecture, sentiment fine-tuning,
head/tail truncation, and dynamic padding. Efficiency measurements belong to
the saved evaluation; this module makes no energy or memory savings claim.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    from src.config import (
        DATA_TRAIN_PATH, MODELS_DIR,
        TINYBERT_MODEL_PATH, TINYBERT_TOKENIZER_PATH, RANDOM_SEED,
    )
except ImportError:
    # Fallback when the file is executed directly as a script (python src/tinybert_model.py)
    from config import (
        DATA_TRAIN_PATH, MODELS_DIR,
        TINYBERT_MODEL_PATH, TINYBERT_TOKENIZER_PATH, RANDOM_SEED,
    )

# Config path is tinybert-internal only; not shared across files
DEFAULT_CONFIG_PATH = f"{MODELS_DIR}/tinybert_config.joblib"

TextLike = Union[str, Sequence[str]]


@dataclass
class TinyBertConfig:
    """Configuration for the Green AI model"""
    # Model selection - TinyBERT for maximum efficiency
    model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
    
    # Sequence length optimization (head+tail strategy)
    max_length: int = 128  # Maximum total input length including special tokens
    head_tokens: int = 63  # First segment of long reviews
    tail_tokens: int = 63  # Last segment of long reviews
    
    # Training efficiency
    batch_size: int = 32
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Green optimizations
    use_fp16: bool = True  # Mixed precision training
    gradient_accumulation_steps: int = 1
    
    # Quantization (applied post-training)
    quantize: bool = True
    
    # GPU optimization
    use_cuda: bool = True  # Will auto-detect GPU availability
    

class IMDbDataset(Dataset):
    """Optimized dataset with text preprocessing"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, config: TinyBertConfig):
        self.texts = [self._clean_text(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove HTML tags and noise"""
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = ' '.join(text.split())
        return text
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Head+Tail truncation strategy
        encoding = self._head_tail_tokenize(text)
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoding
    
    def _head_tail_tokenize(self, text: str):
        """
        Implements head+tail truncation:
        - Tokenize full text
        - If > max_length, take first head_tokens and last tail_tokens
        - This captures intro and conclusion (strongest sentiment signals)
        """
        # Full tokenization without truncation
        tokens = self.tokenizer(
            text,
            truncation=False,
            return_tensors=None,
            add_special_tokens=False,
            verbose=False,
        )
        
        input_ids = tokens['input_ids']
        
        # If short enough, use as-is
        if len(input_ids) <= self.config.max_length - 2:  # -2 for [CLS] and [SEP]
            encoding = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                padding=False,  # Dynamic padding in collator
                return_tensors='pt',
                verbose=False,
            )
        else:
            # Head+Tail strategy
            head = input_ids[:self.config.head_tokens]
            tail = input_ids[-self.config.tail_tokens:]
            combined = head + tail
            
            # Add special tokens
            input_ids = [self.tokenizer.cls_token_id] + combined + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            
            encoding = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        
        # Squeeze to remove batch dimension for single item
        return {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v 
                for k, v in encoding.items()}


class DynamicPaddingCollator(DataCollatorWithPadding):
    """Pad each batch to its longest sequence; energy savings are not measured."""
    def __call__(self, features):
        # Group by similar lengths for maximum efficiency
        batch = super().__call__(features)
        return batch


class TinyBertModel:
    """Fine-tuned TinyBERT wrapper with CPU INT8 inference and head/tail inputs."""

    def __init__(
        self,
        model_path: Optional[str] = TINYBERT_MODEL_PATH,
        config_path: Optional[str] = DEFAULT_CONFIG_PATH,
        load: bool = True,
    ):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.training_mode = False  # BUG FIX: initialize before load() / _build_model()

        # GPU detection and optimization
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"[TinyBertModel] 🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"[TinyBertModel] 💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            print("[TinyBertModel] ⚠️  No GPU detected, using CPU (slower training)")
        
        if load:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}. "
                    "Train it once locally by running `python src/tinybert_model.py`."
                )
            self.load(model_path, config_path)
    
    def _build_model(self):
        """Build a training model online or a checkpoint architecture locally."""
        if self.training_mode:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, num_labels=2,
                problem_type="single_label_classification",
            )
        else:
            architecture_path = Path(MODELS_DIR) / "architecture"
            architecture = AutoConfig.from_pretrained(
                str(architecture_path), local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_config(architecture)

        if self.config.quantize and not self.training_mode:
            self.device = torch.device("cpu")
            model = self._quantize_model(model)
        return model.to(self.device)

    def _quantize_model(self, model):
        """Dynamically quantize linear layers for CPU inference."""
        # Move to CPU for quantization
        model = model.cpu()

        # Mac/ARM (and any platform where PyTorch defaults to 'none') requires
        # an explicit engine before quantize_dynamic; qnnpack works on all CPUs.
        if torch.backends.quantized.engine in ('none', ''):
            torch.backends.quantized.engine = 'qnnpack'

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Keep on CPU for inference (quantized models run best on CPU)
        return quantized_model
    
    def train(self, texts: Sequence[str], labels: Sequence[int], val_texts=None, val_labels=None):
        """Train the model with Green AI optimizations"""
        self.training_mode = True
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = self._build_model()
        
        # Create datasets with dynamic padding
        train_dataset = IMDbDataset(list(texts), list(labels), self.tokenizer, self.config)
        
        if val_texts is not None:
            val_dataset = IMDbDataset(list(val_texts), list(val_labels), self.tokenizer, self.config)
        else:
            val_dataset = None
        
        # Training arguments optimized for efficiency
        training_args = TrainingArguments(
            output_dir=MODELS_DIR,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=100,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="accuracy" if val_dataset else None,
            fp16=self.config.use_fp16 and torch.cuda.is_available(),  # Mixed precision
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Dynamic padding collator
        data_collator = DynamicPaddingCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Trainer with evaluation
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )
        
        # Train
        print("[TinyBertModel] Starting training with Green AI optimizations...")
        print(f"  - Model: {self.config.model_name}")
        print(f"  - Sequence length: {self.config.max_length} (head+tail truncation)")
        print(f"  - Mixed precision: {self.config.use_fp16}")
        print(f"  - Dynamic padding: Enabled")
        
        trainer.train()
        
        # Post-training quantization
        if self.config.quantize:
            print("[TinyBertModel] Applying INT8 quantization for inference...")
            self.model = self._quantize_model(self.model)
            # Update device to CPU after quantization
            self.device = torch.device('cpu')
            print("[TinyBertModel] Model quantized and moved to CPU for efficient inference")
        
        self.training_mode = False
        return trainer
    
    @staticmethod
    def _compute_metrics(eval_pred):
        """Compute accuracy for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions)
        }
    
    def predict(self, text: str) -> int:
        """
        Predict sentiment for a single text
        - Input: str (single text)
        - Output: int (0 or 1)
        
        This matches the evaluation.py interface exactly:
        pred = model.predict(row.text)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded/trained.")
        
        # Clean text
        text = IMDbDataset._clean_text(text)
        
        # Tokenize with head+tail strategy
        encoding = self._head_tail_tokenize_inference(text)
        
        # Convert to tensors
        input_ids = torch.tensor([encoding['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoding['attention_mask']], dtype=torch.long).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=-1).cpu().item()
        
        return int(prediction)
    
    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Predict sentiment for multiple texts (for internal use)
        - Input: list of str
        - Output: list of int
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded/trained.")
        
        # Clean texts
        texts = [IMDbDataset._clean_text(t) for t in texts]
        
        # Tokenize with head+tail strategy
        encodings = []
        for idx, text in enumerate(texts):
            encoding = self._head_tail_tokenize_inference(text)
            encoding["original_idx"] = idx
            encodings.append(encoding)
        
        # Dynamic batching - group similar lengths
        encodings.sort(key=lambda x: len(x['input_ids']))
        
        # Batch inference for efficiency
        all_predictions = [0] * len(encodings)
        batch_size = 32
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(encodings), batch_size):
                batch = encodings[i:i + batch_size]
                
                # Dynamic padding to max in this batch
                max_len = max(len(enc['input_ids']) for enc in batch)
                
                input_ids = []
                attention_masks = []
                
                for enc in batch:
                    ids = enc['input_ids']
                    mask = enc['attention_mask']
                    
                    # Pad to batch max
                    padding_length = max_len - len(ids)
                    ids = ids + [self.tokenizer.pad_token_id] * padding_length
                    mask = mask + [0] * padding_length
                    
                    input_ids.append(ids)
                    attention_masks.append(mask)
                
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
                attention_masks = torch.tensor(attention_masks, dtype=torch.long).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                for enc, pred in zip(batch, predictions.tolist()):
                    all_predictions[enc["original_idx"]] = int(pred)

        return all_predictions
    
    def _head_tail_tokenize_inference(self, text: str):
        """Head+tail tokenization for inference"""
        tokens = self.tokenizer(
            text,
            truncation=False,
            return_tensors=None,
            add_special_tokens=False,
            verbose=False,
        )
        
        input_ids = tokens['input_ids']
        
        if len(input_ids) <= self.config.max_length - 2:
            encoding = self.tokenizer(
                text,
                max_length=self.config.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                verbose=False,
            )
        else:
            head = input_ids[:self.config.head_tokens]
            tail = input_ids[-self.config.tail_tokens:]
            combined = head + tail
            
            input_ids = [self.tokenizer.cls_token_id] + combined + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            
            encoding = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        
        return encoding
    
    def save(self, model_path: str = TINYBERT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        """Save model, tokenizer, and config"""
        if self.model is None:
            raise RuntimeError("Cannot save: model is None.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.config.model_name,
            'quantized': self.config.quantize,
        }, model_path)
        
        # Save architecture so a later state-dict load is independent of caches.
        self.model.config.save_pretrained(Path(MODELS_DIR) / "architecture")

        # Save tokenizer
        self.tokenizer.save_pretrained(TINYBERT_TOKENIZER_PATH)

        # Save config
        joblib.dump(self.config, config_path)

        print(f"[TinyBertModel] Saved model to {model_path}")
        print(f"[TinyBertModel] Saved tokenizer to {TINYBERT_TOKENIZER_PATH}")
        print(f"[TinyBertModel] Saved config to {config_path}")
    
    def load(self, model_path: str = TINYBERT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        """Load model, tokenizer, and config."""
        if config_path and os.path.exists(config_path):
            self.config = joblib.load(config_path)
        else:
            self.config = TinyBertConfig()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TINYBERT_TOKENIZER_PATH, local_files_only=True)

        # Load model
        self.training_mode = False

        # For quantized models, use CPU
        if self.config.quantize:
            self.device = torch.device('cpu')

        # Load checkpoint — supports two artifact formats:
        #   1. State-dict dict  {'model_state_dict': ..., ...}  (produced by save())
        #   2. Full model object  (legacy, produced by archive/tinybert_model_old.py)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: build a fresh quantized model, then populate weights
            self.model = self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Legacy format: the file *is* the model — use it directly
            self.model = checkpoint

        self.model.eval()

        print(f"[TinyBertModel] Loaded model from {model_path}")
        print(f"[TinyBertModel] Running on: {self.device}")

    
    @classmethod
    def train_from_csv(
        cls,
        csv_path: str = DATA_TRAIN_PATH,
        model_path: str = TINYBERT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        text_col: str = "text",
        label_col: str = "label",
        config: Optional[TinyBertConfig] = None,
        val_size: float = 0.2,
        random_state: int = RANDOM_SEED,
    ) -> "TinyBertModel":
        """
        Complete training pipeline from CSV
        - Loads data
        - Splits train/val
        - Trains with Green AI optimizations
        - Evaluates
        - Saves model
        """
        df = pd.read_csv(csv_path)
        assert {text_col, label_col}.issubset(df.columns), \
            f"CSV must contain columns: {text_col}, {label_col}"
        
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=val_size,
            random_state=random_state,
            stratify=labels,
        )
        
        print(f"[TinyBertModel] Training samples: {len(X_train)}")
        print(f"[TinyBertModel] Validation samples: {len(X_val)}")
        
        # Initialize and train
        model = cls(load=False)
        model.config = config or TinyBertConfig()
        model.train(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        print("\n[TinyBertModel] Final Validation Evaluation:")
        val_preds = model.predict_batch(X_val)
        acc = accuracy_score(y_val, val_preds)
        print(f"Validation Accuracy: {acc:.4f}")
        print(classification_report(y_val, val_preds, target_names=['Negative', 'Positive']))
        
        print("\n[TinyBertModel] Configuration used:")
        print(f"  Model: {model.config.model_name}")
        print(f"  Maximum input length: {model.config.max_length}")
        print(f"  Dynamic INT8 linear layers: {model.config.quantize}")
        print(f"  Validation accuracy: {acc:.2%}")

        # Save
        model.save(model_path, config_path)
        
        return model


# -------------------------------------------------------------------------
# Entry point: Train the model locally
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("GREEN AI MODEL - Training Pipeline")
    print("Training a fresh sentiment classifier from pretrained TinyBERT")
    print("=" * 80)
    
    # Train with optimized config
    config = TinyBertConfig(
        model_name="huawei-noah/TinyBERT_General_4L_312D",
        max_length=128,
        head_tokens=63,
        tail_tokens=63,
        batch_size=32,
        learning_rate=3e-5,
        num_epochs=3,
        use_fp16=True,
        quantize=True,
    )
    
    model = TinyBertModel.train_from_csv(config=config)
    
    print("\n" + "=" * 80)
    print("Training Complete! Model ready for evaluation.")
    print("=" * 80)
