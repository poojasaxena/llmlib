#!/usr/bin/env python3

"""
Standalone data generation runner
Allows running individual data generation components separately
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run individual data generation components")
    parser.add_argument("component", choices=["advanced", "webscrape", "augment", "all"],
                        help="Which component to run")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--source", type=Path, help="Source file for augmentation/advanced synthetic")
    parser.add_argument("--count", type=int, default=10000, help="Number of items to generate")
    
    args = parser.parse_args()
    
    if args.component in ["advanced", "all"]:
        if not args.source or not args.source.exists():
            print("❌ Advanced synthetic generation requires --source file")
            return 1
            
        print("🔄 Running advanced synthetic expansion...")
        try:
            from llmlib.data.advanced_synth_expand import main as advanced_main
            # Set up sys.argv for the script
            old_argv = sys.argv
            sys.argv = ["advanced_synth_expand.py", str(args.output_dir), str(args.source), str(args.count)]
            advanced_main()
            sys.argv = old_argv
            print("✅ Advanced synthetic expansion completed")
        except Exception as e:
            print(f"❌ Advanced synthetic expansion failed: {e}")
            return 1
    
    if args.component in ["webscrape", "all"]:
        print("🔄 Running web scraping...")
        try:
            from llmlib.data.web_scrape_elephants import main as webscrape_main
            old_argv = sys.argv
            sys.argv = ["web_scrape_elephants.py", str(args.output_dir), str(args.count)]
            webscrape_main()
            sys.argv = old_argv
            print("✅ Web scraping completed")
        except Exception as e:
            print(f"❌ Web scraping failed: {e}")
            return 1
    
    if args.component in ["augment", "all"]:
        if not args.source or not args.source.exists():
            print("❌ Data augmentation requires --source file")
            return 1
            
        print("🔄 Running data augmentation...")
        try:
            from llmlib.data.augment_corpus import main as augment_main
            old_argv = sys.argv
            sys.argv = ["augment_corpus.py", str(args.output_dir), str(args.source)]
            augment_main()
            sys.argv = old_argv
            print("✅ Data augmentation completed")
        except Exception as e:
            print(f"❌ Data augmentation failed: {e}")
            return 1
    
    print("🎉 Data generation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
