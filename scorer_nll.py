# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CECalculatorConfig:
    model_id: str
    batch_size: int
    surrogate_attack_prompt: str

class PrefixCECalculator:
    def __init__(self, config: CECalculatorConfig):
        """
        Initialize the Cross-Entropy Calculator.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Loading model: {config.model_id} for CE calculation")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Load model
        # using device_map="auto" to handle large models automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

    def calculate_ce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Cross-Entropy (NLL) for the prefixes in the dataframe.
        """
        logging.info(f"Calculating CE for {len(df)} prefixes")
        
        # Ensure we have the required columns
        if 'goal' not in df.columns or 'prefix' not in df.columns:
            raise ValueError("Dataframe must contain 'goal' and 'prefix' columns")

        # Prepare texts
        # We construct the input as: <goal> <prefix> <surrogate>
        # The loss is usually calculated on the surrogate or the prefix depending on the strategy.
        # Given the config, we append the surrogate if it exists.
        texts = []
        surrogate = self.config.surrogate_attack_prompt if self.config.surrogate_attack_prompt else ""
        
        for _, row in df.iterrows():
            # Basic formatting
            prompt = f"{row['goal']} {row['prefix']}"
            if surrogate:
                prompt += f" {surrogate}"
            texts.append(prompt)

        all_losses = []
        batch_size = self.config.batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing NLL"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                # --- FIX: Move inputs to the same device as the model ---
                # This handles the "Expected all tensors to be on the same device" error
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs, use_cache=False)
                logits = outputs.logits
                
                # Calculate loss
                # Shift logits and labels to align prediction with target
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                # Calculate Cross Entropy per sample
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Reshape back to [batch_size, seq_len]
                loss = loss.view(shift_labels.size(0), shift_labels.size(1))
                
                # Average loss over non-padded tokens (perplexity/NLL score)
                # We use attention_mask to ignore padding
                mask = inputs["attention_mask"][..., 1:].contiguous()
                
                # Sum loss per sequence and divide by number of non-pad tokens
                sequence_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
                
                all_losses.extend(sequence_loss.float().cpu().numpy().tolist())

        # Add results to dataframe
        df['prefix_nll'] = all_losses
        return df