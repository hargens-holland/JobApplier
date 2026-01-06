"""Simple test: Load job.txt and print the extracted output."""

import time
from src.loader import load_job

def test_job_parsing():
    """Load job.txt and print extracted information."""
    print("=" * 60)
    print("JOB PARSING TEST")
    print("=" * 60)
    print("Loading job from data/job.txt...\n")
    
    start_time = time.time()
    
    # Load job (will use cached model if available)
    job = load_job(use_llm=True)
    
    load_time = time.time() - start_time
    
    # Print results
    print("=" * 60)
    print("EXTRACTED INFORMATION")
    print("=" * 60)
    print(f"Title: {job.title}")
    print(f"Company: {job.company}")
    print(f"Location: {job.location}")
    print(f"\nSkills ({len(job.skills)}):")
    for skill in job.skills:
        print(f"  • {skill}")
    print(f"\nDescription:")
    print(f"  {job.description[:300]}..." if len(job.description) > 300 else f"  {job.description}")
    print(f"\nExtraction Method: {job.metadata.get('extraction_method')}")
    print(f"Load Time: {load_time:.2f} seconds")
    print("=" * 60)
    
    return job

if __name__ == "__main__":
    test_job_parsing()

