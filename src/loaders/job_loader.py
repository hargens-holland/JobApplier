"""Loader for job posting data using local LLM for extraction."""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

from ..models.job import JobPosting


class JobLoader:
    """Loads job posting data from files and returns domain objects."""
    
    # Class-level cache for models (shared across all instances)
    _model_cache: Dict[str, tuple] = {}  # {model_name: (model, tokenizer)}
    
    def __init__(self, use_llm: bool = True, model_name: Optional[str] = None):
        """
        Initialize JobLoader.
        
        Args:
            use_llm: If True, use local LLM for extraction (default: True)
            model_name: Hugging Face model name. Defaults to a small efficient model.
        """
        self.use_llm = use_llm
        # Model options for JSON extraction (smallest to largest):
        # 
        # RECOMMENDED: "Qwen/Qwen2-0.5B-Instruct" (~1GB)
        #   - Smallest viable size for reliable JSON extraction
        #   - Excellent instruction following
        #   - Fast on CPU (~2-5 sec per extraction)
        #
        # Alternatives:
        # - "microsoft/phi-1_5" (~2.6GB, better quality but larger)
        # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (~2.2GB, decent)
        # - "microsoft/Phi-3-mini-4k-instruct" (~7GB, best quality)
        # 
        # Models are cached in: ~/.cache/huggingface/hub/
        # To use a different location, set HF_HOME environment variable
        self.model_name = model_name or "Qwen/Qwen2-0.5B-Instruct"
        self._model = None
        self._tokenizer = None
        
        if use_llm:
            self._ensure_model_loaded()
    
    @classmethod
    def establish_connection(cls, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> bool:
        """
        Establish connection to LLM model (loads model into cache).
        
        Args:
            model_name: Hugging Face model name
            
        Returns:
            True if connection successful, False otherwise
        """
        if model_name in cls._model_cache:
            # Model already in memory - no need to reload
            return True
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            print(f"✗ Error: transformers not installed: {e}")
            print("   Install with: pip install transformers torch")
            return False
        
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            print("   ℹ torch not available, using CPU only")
        
        try:
            # Check if model exists on disk first
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_dir_name = f"models--{model_name.replace('/', '--')}"
            model_path = os.path.join(cache_dir, model_dir_name)
            
            if os.path.exists(model_path):
                print(f"Loading model from disk cache: {model_name}...")
                print("   (Model already downloaded - loading into memory, ~15-30 seconds)")
            else:
                print(f"Downloading and loading model: {model_name}...")
                print("   (First time - will download ~1GB, then load into memory)")
            
            print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("   ✓ Tokenizer loaded")
            
            print("   Loading model into memory (this takes ~15-30 seconds)...")
            
            # Check if accelerate is available (needed for low_cpu_mem_usage)
            try:
                import accelerate
                has_accelerate = True
            except ImportError:
                has_accelerate = False
                print("   ℹ accelerate not installed (optional, but recommended for faster loading)")
            
            if has_torch and torch.cuda.is_available():
                if has_accelerate:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    )
                print(f"   ✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                if has_accelerate:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=True
                    )
                else:
                    # Without accelerate, don't use low_cpu_mem_usage
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name
                    )
                print("   ✓ Model loaded on CPU")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Cache the model
            cls._model_cache[model_name] = (model, tokenizer)
            print(f"✓ Connection established! Model '{model_name}' is ready.")
            return True
            
        except ImportError as e:
            error_msg = str(e)
            if "accelerate" in error_msg.lower():
                print("✗ Error: accelerate package required")
                print("   Install with: pip install accelerate")
                print("   (This is needed for efficient model loading)")
            else:
                print(f"✗ Error: Missing dependency: {e}")
                print("   Install required packages: pip install transformers torch accelerate")
            return False
        except Exception as e:
            import traceback
            print(f"✗ Error: Could not establish connection")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            print(f"\n   Full traceback:")
            traceback.print_exc()
            return False
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (use cached version if available)."""
        # Check if model is already in memory cache
        if self.model_name in JobLoader._model_cache:
            # Use cached model (already in memory - instant!)
            self._model, self._tokenizer = JobLoader._model_cache[self.model_name]
        else:
            # Model not in memory - load it
            # establish_connection will load from disk cache (faster than download)
            if not self.establish_connection(self.model_name):
                raise RuntimeError("Failed to load LLM model. LLM extraction is required.")
            else:
                # Get from cache after loading
                self._model, self._tokenizer = JobLoader._model_cache[self.model_name]
    
    def load_from_txt(self, path: str) -> JobPosting:
        """
        Load job posting from text file.
        
        Args:
            path: Path to the text file
            
        Returns:
            JobPosting domain object
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Job file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use LLM extraction (required)
        if not self.use_llm or self._model is None:
            raise RuntimeError("LLM extraction is required. Set use_llm=True and ensure model is loaded.")
        
        try:
            extracted_data = self._extract_with_llm(content)
            title = extracted_data.get('title', 'Job Position')
            company = extracted_data.get('company', 'Company')
            location = extracted_data.get('location', 'Not specified')
            skills = extracted_data.get('skills', [])
            description = extracted_data.get('description', content.strip())
        except Exception as e:
            raise RuntimeError(f"LLM extraction failed: {e}. Please check the job posting format and try again.")
        
        # Metadata
        metadata = {
            'filename': path_obj.name,
            'source': 'text_file',
            'extraction_method': 'llm'
        }
        
        return JobPosting(
            title=title,
            company=company,
            description=description,
            skills=skills,
            location=location,
            metadata=metadata
        )
    
    def _extract_with_llm(self, job_text: str) -> Dict:
        """
        Extract structured data from job posting using local LLM.
        
        Args:
            job_text: Raw job posting text
            
        Returns:
            Dictionary with extracted fields
        """
        # Use full text (most job postings are under 5000 chars, which fits in context)
        # Only limit if extremely long to avoid token issues
        job_text_limited = job_text[:5000] if len(job_text) > 5000 else job_text
        
        prompt = f"""Extract job information as JSON. Return ONLY valid JSON with these fields:

{{
  "title": "job title",
  "company": "company name",
  "location": "location",
  "skills": ["list", "ALL", "technical", "skills", "mentioned"],
  "description": "brief 2-3 sentence summary"
}}

SKILLS: Extract EVERY technical skill, language, tool, framework, or technology mentioned. Examples:
- Languages: Python, SQL, Java, JavaScript, C++, R (extract ALL mentioned)
- Tools: Git, Docker, AWS, BI tools, ETL tools
- Technologies: Machine Learning, LLMs, Deep Learning, Reinforcement Learning
- Concepts: Statistics, Data Analysis, Visualization, Experimentation, APIs, Databases

IMPORTANT: If text says "Python, SQL" extract BOTH as separate items: ["Python", "SQL"]
Read ALL sections: Must-Have, Nice-to-Have, What You'll Work On, requirements, etc.

Job Posting:
{job_text_limited}

Return ONLY the JSON object, no markdown, no explanations:"""

        try:
            # Format prompt for chat model
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from job postings. Always return valid JSON only, no other text."},
                {"role": "user", "content": prompt}
            ]
            
            # Format for chat template (if available)
            try:
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except (AttributeError, TypeError):
                # Fallback if chat template not available
                formatted_prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            
            # Tokenize
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            try:
                import torch
                if hasattr(self._model, 'device'):
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                
                # Generate (optimized for speed)
                with torch.no_grad():
                    # Suppress warnings about sampling params when do_sample=False
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
                        warnings.filterwarnings("ignore", message=".*do_sample.*top_p.*")
                        warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*")
                        
                        gen_kwargs = {
                            "max_new_tokens": 800,  # Increased to allow comprehensive skill lists
                            "do_sample": False,
                            "pad_token_id": self._tokenizer.eos_token_id
                        }
                        
                        outputs = self._model.generate(**inputs, **gen_kwargs)
            except ImportError:
                # Fallback if torch not available
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
                    warnings.filterwarnings("ignore", message=".*do_sample.*top_p.*")
                    warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*")
                    
                    gen_kwargs = {
                        "max_new_tokens": 800,  # Increased to allow comprehensive skill lists
                        "do_sample": False,
                        "pad_token_id": self._tokenizer.eos_token_id
                    }
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            
            # Decode
            response = self._tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract JSON from response
            json_text = self._extract_json_from_text(response)
            result = json.loads(json_text)
            
            # Post-process: Add critical skills that are mentioned in job text but missing from LLM extraction
            # The 0.5B model sometimes misses specific skills like Python/SQL
            skills = result.get('skills', [])
            skills_lower = [s.lower() for s in skills]
            
            # Common programming languages and tools to check for
            critical_skills = {
                'python': 'Python',
                'sql': 'SQL',
                'java': 'Java',
                'javascript': 'JavaScript',
                'typescript': 'TypeScript',
                'c++': 'C++',
                'c#': 'C#',
                'r': 'R',
                'go': 'Go',
                'rust': 'Rust',
                'etl': 'ETL',
                'aws': 'AWS',
                'docker': 'Docker',
                'kubernetes': 'Kubernetes',
                'git': 'Git',
            }
            
            # Post-process: If a critical skill is in job_text but NOT in JSON → add it
            # Use word boundaries to avoid false matches (e.g., "rust" in "trust", "r" in "for")
            job_text_lower = job_text_limited.lower()
            added_skills = []
            
            for keyword, skill_name in critical_skills.items():
                # Use word boundary regex to match whole words only
                # Special handling for multi-character keywords like "c++", "c#"
                if keyword in ['c++', 'c#']:
                    # For these, escape special chars and use word boundary
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                else:
                    # For single words, use word boundaries
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                
                # Check if keyword found as whole word in job text AND skill not already in extracted skills
                if re.search(pattern, job_text_lower) and skill_name.lower() not in skills_lower:
                    skills.append(skill_name)
                    added_skills.append(skill_name)
            
            if added_skills:
                print(f"[DEBUG] ✓ Added missing skills from job text: {', '.join(added_skills)}")
            
            result['skills'] = skills
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"LLM extraction failed: {str(e)}")
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from LLM response text."""
        # Clean up the text
        text = text.strip()
        
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object (handles nested objects)
        # Match from first { to matching }
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
        
        # Fallback: try regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        raise ValueError("Could not extract valid JSON from response")

