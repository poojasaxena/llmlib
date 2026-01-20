#!/usr/bin/env python3

"""
Test script for the enhanced data pipeline
Creates a minimal test environment to verify the pipeline works
"""

import os
import tempfile
from pathlib import Path
import shutil


def create_test_environment():
    """Create a temporary test environment with sample data."""
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="llmlib_test_"))
    print(f"🧪 Created test environment: {temp_dir}")
    
    # Set up directory structure
    raw_dir = temp_dir / "llm/mixed_text/raw"
    out_dir = temp_dir / "llm/mixed_text/out"
    raw_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    
    # Create sample raw data
    sample_data = [
        "Elephants are magnificent creatures with incredible intelligence.",
        "African elephants have larger ears than Asian elephants.",
        "Elephants can live up to 70 years in the wild.",
        "Baby elephants are called calves and weigh about 250 pounds at birth.",
        "Elephants use their trunks for breathing, drinking, and communication.",
        "The elephant's trunk contains over 40,000 muscles.",
        "Elephants are herbivores and can eat up to 300 pounds of vegetation per day.",
        "Elephants have excellent memories and can remember other elephants for decades.",
        "Matriarchal herds are led by the oldest female elephant.",
        "Elephants show empathy and mourn their dead.",
    ]
    
    # Write sample files
    (raw_dir / "sample1.txt").write_text("\n".join(sample_data[:5]) + "\n")
    (raw_dir / "sample2.txt").write_text("\n".join(sample_data[5:]) + "\n")
    
    print(f"📄 Created sample raw data in: {raw_dir}")
    print(f"   sample1.txt: {len(sample_data[:5])} lines")
    print(f"   sample2.txt: {len(sample_data[5:])} lines")
    
    return temp_dir


def test_pipeline():
    """Test the enhanced pipeline with sample data."""
    
    # Create test environment
    test_dir = create_test_environment()
    
    try:
        # Set environment variable for the pipeline
        os.environ["GLOBAL_DATASETS_DIR"] = str(test_dir)
        
        # Import and run the pipeline
        from llmlib.data.pipeline.run_full_data_pipeline import main
        
        print("\n🚀 Running enhanced data pipeline...")
        main()
        
        # Check results
        out_dir = test_dir / "llm/mixed_text/out"
        if out_dir.exists():
            print(f"\n✅ Pipeline completed! Results in: {out_dir}")
            
            # List final files
            for file in out_dir.iterdir():
                if file.is_file():
                    size = file.stat().st_size
                    lines = sum(1 for _ in file.open('r', encoding='utf-8')) if file.suffix == '.txt' else 0
                    print(f"   📄 {file.name}: {lines} lines ({size} bytes)")
        else:
            print("❌ Pipeline output directory not found")
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print(f"\n🗑️  Cleaning up test environment: {test_dir}")
        shutil.rmtree(test_dir)
        
        # Restore environment
        if "GLOBAL_DATASETS_DIR" in os.environ:
            del os.environ["GLOBAL_DATASETS_DIR"]


def main():
    """Main entry point for the test."""
    test_pipeline()


if __name__ == "__main__":
    main()
