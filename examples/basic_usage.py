#!/usr/bin/env python3
"""
Basic usage example for Granite Docling 258M

This script demonstrates how to use the Granite Docling model for
document conversion tasks.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from granite_docling import GraniteDocling, download_models


def main():
    """Main example function."""
    print("Granite Docling 258M - Basic Usage Example")
    print("=" * 50)

    try:
        # Initialize the Granite Docling processor
        print("Initializing Granite Docling...")
        granite = GraniteDocling()

        # Example 1: Convert a PDF from URL
        print("\nExample 1: Converting PDF from URL")
        pdf_url = "https://arxiv.org/pdf/2206.01062"  # Example academic paper

        try:
            result = granite.convert_document(pdf_url)
            print(f"✅ Successfully converted document!")
            print(f"   Pages: {result['pages']}")
            print(f"   Content length: {len(result['content'])} characters")
            print(f"   First 200 characters: {result['content'][:200]}...")

            # Save to file
            output_path = "outputs/converted_paper.md"
            granite.convert_to_file(pdf_url, output_path)
            print(f"   Saved to: {output_path}")

        except Exception as e:
            print(f"❌ Failed to convert PDF: {e}")

        # Example 2: Batch conversion (if you have local files)
        print("\nExample 2: Batch conversion")
        print("ℹ️  To use batch conversion, place PDF files in 'sample_docs/' directory")

        sample_docs_dir = Path("sample_docs")
        if sample_docs_dir.exists():
            pdf_files = list(sample_docs_dir.glob("*.pdf"))
            if pdf_files:
                print(f"Found {len(pdf_files)} PDF files for batch conversion")
                results = granite.batch_convert(
                    sources=[str(f) for f in pdf_files],
                    output_dir="outputs/batch_results"
                )

                successful = sum(1 for r in results if 'error' not in r)
                print(f"✅ Successfully converted {successful}/{len(results)} documents")
            else:
                print("No PDF files found in sample_docs/")
        else:
            print("sample_docs/ directory not found")

        print("\n🎉 Example completed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def setup_directories():
    """Create necessary directories for examples."""
    directories = ["outputs", "sample_docs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")


if __name__ == "__main__":
    print("Setting up directories...")
    setup_directories()

    print("\nChecking if models are downloaded...")
    try:
        download_models()
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("You may need to download models manually using:")
        print("docling-tools models download-hf-repo ibm-granite/granite-docling-258M")

    main()