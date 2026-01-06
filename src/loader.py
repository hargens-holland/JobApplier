"""Main loader module for loading resume and job data."""

from pathlib import Path
from .loaders.resume_loader import ResumeLoader
from .loaders.job_loader import JobLoader


def load_resume(resume_path: str = None) -> 'Resume':
    """
    Load resume from JSON file.
    
    Args:
        resume_path: Path to resume JSON file. Defaults to data/resume.json
        
    Returns:
        Resume domain object
    """
    if resume_path is None:
        resume_path = Path(__file__).parent.parent / 'data' / 'resume.json'
    
    loader = ResumeLoader()
    return loader.load_from_json(str(resume_path))


def load_job(job_path: str = None, use_llm: bool = True) -> 'JobPosting':
    """
    Load job posting from text file.
    
    Args:
        job_path: Path to job text file. Defaults to data/job.txt
        use_llm: If True, use LLM for extraction (default: True)
        
    Returns:
        JobPosting domain object
    """
    if job_path is None:
        job_path = Path(__file__).parent.parent / 'data' / 'job.txt'
    
    loader = JobLoader(use_llm=use_llm)
    return loader.load_from_txt(str(job_path))


def load_all(resume_path: str = None, job_path: str = None):
    """
    Load both resume and job posting.
    
    Args:
        resume_path: Path to resume JSON file. Defaults to data/resume.json
        job_path: Path to job text file. Defaults to data/job.txt
        
    Returns:
        Tuple of (Resume, JobPosting)
    """
    resume = load_resume(resume_path)
    job = load_job(job_path)
    return resume, job

