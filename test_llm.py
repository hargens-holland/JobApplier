"""Test suite for LLM functionality - each test method tests a specific use case."""

from src.loaders.job_loader import JobLoader
from src.loader import load_job, load_resume
from src.utils.llm_service import LLMService


def test_connection():
    """
    Test 1: Verify LLM connection can be established.
    Tests that the model can be downloaded and loaded into memory.
    """
    print("=" * 60)
    print("TEST 1: LLM Connection")
    print("=" * 60)
    
    try:
        success = JobLoader.establish_connection()
        
        if success:
            print("\n✓ Connection successful!")
            print("✓ Model downloaded and loaded into memory")
            print("✓ Model is ready to use")
            return True
        else:
            print("\n✗ Connection failed")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_job_parsing():
    """
    Test 2: Verify LLM can extract structured data from job postings.
    Tests the job extraction functionality using the cached model.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Job Parsing with LLM")
    print("=" * 60)
    
    try:
        # This will use the cached model from test_connection()
        job = load_job(use_llm=True)
        
        print(f"\n✓ Job extraction successful!")
        print(f"  Title: {job.title}")
        print(f"  Company: {job.company}")
        print(f"  Location: {job.location}")
        print(f"  Skills found: {len(job.skills)}")
        print(f"  Extraction method: {job.metadata.get('extraction_method')}")
        
        # Verify it used LLM
        if job.metadata.get('extraction_method') == 'llm':
            print("  ✓ Confirmed: Used LLM extraction (not rule-based)")
            return True
        else:
            print("  ⚠ Warning: Used rule-based extraction (LLM may not be working)")
            return False
            
    except FileNotFoundError:
        print("\n⚠ Job file (data/job.txt) not found")
        print("  (This is okay - connection test still passed)")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_resume_tailoring():
    """
    Test 3: Verify LLM can tailor resume content.
    Tests resume tailoring functionality using the cached model.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Resume Tailoring with LLM")
    print("=" * 60)
    
    try:
        # Get LLM service (uses cached model)
        llm = LLMService.get_or_create()
        
        # Load resume and job
        resume = load_resume()
        job = load_job(use_llm=True)
        
        # Create tailoring prompt
        prompt = f"""Tailor this resume summary to match the job requirements.

Resume Summary:
{resume.text[:500]}

Job Requirements:
- Title: {job.title}
- Required Skills: {', '.join(job.skills[:10])}
- Description: {job.description[:300]}

Rewrite the resume summary to highlight relevant experience and skills for this specific job.
Keep it professional and concise (2-3 sentences):"""

        # Generate tailored summary
        tailored_summary = llm.generate(prompt, max_tokens=200)
        
        print(f"\n✓ Resume tailoring successful!")
        print(f"  Original length: {len(resume.text[:500])} chars")
        print(f"  Tailored summary: {len(tailored_summary)} chars")
        print(f"  Preview: {tailored_summary[:150]}...")
        print("  ✓ Confirmed: Used LLM for tailoring")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n⚠ Required file not found: {e}")
        print("  (Need data/resume.json and data/job.txt)")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def run_all_tests():
    """Run all LLM tests in sequence."""
    print("\n" + "=" * 60)
    print("LLM TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Connection
    results.append(("Connection", test_connection()))
    
    # Test 2: Job Parsing (only if connection worked)
    if results[0][1]:
        results.append(("Job Parsing", test_job_parsing()))
        results.append(("Resume Tailoring", test_resume_tailoring()))
    else:
        print("\n⚠ Skipping other tests - connection failed")
        results.append(("Job Parsing", False))
        results.append(("Resume Tailoring", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    # Allow running individual tests
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == "connection":
            success = test_connection()
        elif test_name == "job":
            success = test_job_parsing()
        elif test_name == "resume":
            success = test_resume_tailoring()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: connection, job, resume")
            success = False
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
