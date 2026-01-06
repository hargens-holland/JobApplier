"""Shared LLM service for reuse across different tasks."""

import json
import re
from typing import Dict, List, Optional, Any


class LLMService:
    """
    Shared LLM service that loads the model once and can be reused
    for multiple tasks: job extraction, resume tailoring, etc.
    """
    
    # Class-level cache - shared across all instances
    _model_cache: Dict[str, tuple] = {}  # {model_name: (model, tokenizer)}
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize LLM service.
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._ensure_model_loaded()
    
    @classmethod
    def get_or_create(cls, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> 'LLMService':
        """
        Get existing service instance or create new one.
        Reuses the same model cache.
        """
        return cls(model_name)
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (use cached version if available)."""
        # Check JobLoader cache first (they share the same model)
        from ..loaders.job_loader import JobLoader
        if self.model_name in JobLoader._model_cache:
            self._model, self._tokenizer = JobLoader._model_cache[self.model_name]
            # Also cache in LLMService cache for consistency
            LLMService._model_cache[self.model_name] = (self._model, self._tokenizer)
        elif self.model_name in LLMService._model_cache:
            self._model, self._tokenizer = LLMService._model_cache[self.model_name]
        else:
            self._load_model()
    
    def _load_model(self):
        """Load the model into cache."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers torch accelerate")
        
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        
        try:
            import accelerate
            has_accelerate = True
        except ImportError:
            has_accelerate = False
        
        print(f"Loading model: {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if has_torch and torch.cuda.is_available():
            if has_accelerate:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                )
        else:
            if has_accelerate:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Cache the model
        LLMService._model_cache[self.model_name] = (model, tokenizer)
        self._model, self._tokenizer = model, tokenizer
        print(f"✓ Model '{self.model_name}' loaded and cached.")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Format prompt for chat
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError):
            formatted = prompt
        
        inputs = self._tokenizer(formatted, return_tensors="pt")
        try:
            import torch
            if hasattr(self._model, 'device'):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id
                )
        except ImportError:
            # Fallback if torch not available
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def extract_json(self, prompt: str) -> Dict[str, Any]:
        """
        Extract structured JSON from a prompt.
        
        Args:
            prompt: Prompt that should result in JSON output
            
        Returns:
            Parsed JSON dictionary
        """
        response = self.generate(prompt, max_tokens=512, temperature=0.1)
        
        # Extract JSON from response
        json_text = self._extract_json_from_text(response)
        return json.loads(json_text)
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text."""
        text = text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        brace_count = 0
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        
        raise ValueError("Could not extract valid JSON from response")
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache (free memory)."""
        cls._model_cache.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

