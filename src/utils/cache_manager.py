"""Utility to manage model cache and disk usage."""

import os
from pathlib import Path
from typing import Dict, Optional


def get_cache_location() -> Path:
    """Get the Hugging Face cache location."""
    cache_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "huggingface" / "hub"


def get_cache_size() -> Dict[str, any]:
    """
    Get information about model cache size.
    
    Returns:
        Dictionary with cache info:
        - location: Path to cache
        - total_size_gb: Total size in GB
        - model_count: Number of models
        - models: List of model info
    """
    cache_path = get_cache_location()
    
    if not cache_path.exists():
        return {
            "location": str(cache_path),
            "total_size_gb": 0,
            "model_count": 0,
            "models": []
        }
    
    models = []
    total_size = 0
    
    # Find all model directories
    for model_dir in cache_path.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("models--"):
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            
            # Calculate size
            model_size = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            )
            
            total_size += model_size
            models.append({
                "name": model_name,
                "size_gb": round(model_size / (1024**3), 2),
                "path": str(model_dir)
            })
    
    return {
        "location": str(cache_path),
        "total_size_gb": round(total_size / (1024**3), 2),
        "model_count": len(models),
        "models": sorted(models, key=lambda x: x["size_gb"], reverse=True)
    }


def print_cache_info():
    """Print cache information in a readable format."""
    info = get_cache_size()
    
    print("=" * 60)
    print("MODEL CACHE INFORMATION")
    print("=" * 60)
    print(f"Location: {info['location']}")
    print(f"Total Size: {info['total_size_gb']} GB")
    print(f"Number of Models: {info['model_count']}")
    print()
    
    if info['models']:
        print("Models in cache:")
        for model in info['models']:
            print(f"  • {model['name']}: {model['size_gb']} GB")
    else:
        print("No models cached yet.")
    
    print("=" * 60)


def clear_model_cache(model_name: Optional[str] = None) -> bool:
    """
    Clear model cache.
    
    Args:
        model_name: Specific model to clear (e.g., "Qwen/Qwen2-0.5B-Instruct")
                   If None, clears all models.
    
    Returns:
        True if successful
    """
    cache_path = get_cache_location()
    
    if not cache_path.exists():
        print("No cache to clear.")
        return True
    
    if model_name:
        # Clear specific model
        model_dir_name = f"models--{model_name.replace('/', '--')}"
        model_path = cache_path / model_dir_name
        
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path)
            print(f"✓ Cleared cache for: {model_name}")
            return True
        else:
            print(f"Model not found in cache: {model_name}")
            return False
    else:
        # Clear all models
        import shutil
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            print("✓ Cleared all model cache")
            return True
    
    return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        if len(sys.argv) > 2:
            clear_model_cache(sys.argv[2])
        else:
            response = input("Clear ALL model cache? (yes/no): ")
            if response.lower() == "yes":
                clear_model_cache()
            else:
                print("Cancelled.")
    else:
        print_cache_info()

