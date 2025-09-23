#!/usr/bin/env python3
"""
Basic usage example for Granite Docling 258M

This script demonstrates how to use the Granite Docling model for
document conversion tasks with proper error handling and validation.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import urllib.parse
import urllib.request

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from granite_docling import GraniteDocling, download_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """
    Validate that a URL is properly formatted and accessible.

    Args:
        url: The URL to validate

    Returns:
        bool: True if URL is valid and accessible, False otherwise
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False

        # Try to open the URL to check if it's accessible
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        logger.error(f"URL validation failed for {url}: {e}")
        return False


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable.

    Args:
        file_path: Path to the file to validate

    Returns:
        bool: True if file exists and is readable, False otherwise
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False

        if not os.access(path, os.R_OK):
            logger.error(f"File is not readable: {file_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"File validation failed for {file_path}: {e}")
        return False


def setup_directories() -> None:
    """Create necessary directories for examples."""
    directories = ["outputs", "sample_docs"]

    for directory in directories:
        try:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
            print(f"[CREATE] Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            print(f"[ERROR] Failed to create directory {directory}: {e}")
            raise


def demonstrate_single_document_conversion(granite: GraniteDocling) -> bool:
    """
    Demonstrate converting a single document from URL.

    Args:
        granite: Initialized GraniteDocling instance

    Returns:
        bool: True if conversion succeeded, False otherwise
    """
    print("\n" + "=" * 60)
    print("Example 1: Converting PDF from URL")
    print("=" * 60)

    # Use a reliable academic paper URL
    pdf_url = "https://arxiv.org/pdf/2206.01062"  # Example academic paper

    try:
        # Validate URL first
        print(f"[SOURCE] Source URL: {pdf_url}")
        print("[VALIDATE] Validating URL accessibility...")

        if not validate_url(pdf_url):
            print(f"[ERROR] URL is not accessible: {pdf_url}")
            print("[INFO] This might be due to network issues or URL changes.")
            return False

        print("[PASS] URL is accessible")
        print("[CONVERT] Starting document conversion...")

        # Convert the document
        result = granite.convert_document(pdf_url)

        # Validate the result
        if not result or 'content' not in result:
            print("[ERROR] Conversion returned invalid result")
            return False

        if not result['content'] or len(result['content'].strip()) == 0:
            print("[ERROR] Conversion returned empty content")
            return False

        # Display conversion results
        print("[SUCCESS] Successfully converted document!")
        print(f"   [INFO] Pages: {result.get('pages', 'Unknown')}")
        print(f"   [INFO] Content length: {len(result['content'])} characters")
        print(f"   [PREVIEW] First 200 characters: {result['content'][:200]}...")

        # Save to file with proper error handling
        try:
            output_path = Path("outputs") / "converted_paper.md"
            granite.convert_to_file(pdf_url, output_path)
            print(f"   [SAVE] Saved to: {output_path}")

            # Validate that the file was actually created and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"   [PASS] Output file validated: {output_path.stat().st_size} bytes")
            else:
                print(f"   [WARN] Output file validation failed")
                return False

        except Exception as save_error:
            logger.error(f"Failed to save converted document: {save_error}")
            print(f"   [ERROR] Failed to save document: {save_error}")
            return False

        return True

    except Exception as e:
        logger.error(f"Document conversion failed: {e}")
        print(f"[ERROR] Failed to convert PDF: {e}")
        print("\nTroubleshooting suggestions:")
        print("  - Check your internet connection")
        print("  - Verify the URL is still valid")
        print("  - Ensure you have sufficient disk space")
        print("  - Check that all dependencies are properly installed")
        return False


def demonstrate_batch_processing(granite: GraniteDocling) -> bool:
    """
    Demonstrate batch processing of multiple documents.

    Args:
        granite: Initialized GraniteDocling instance

    Returns:
        bool: True if batch processing demonstration succeeded, False otherwise
    """
    print("\n" + "=" * 60)
    print("Example 2: Batch Document Processing")
    print("=" * 60)

    sample_docs_dir = Path("sample_docs")

    try:
        # Check if sample_docs directory exists
        if not sample_docs_dir.exists():
            print(f"[INFO] Sample documents directory not found: {sample_docs_dir}")
            print("[INFO] Creating example setup for batch processing...")

            # Create some example file references (not actual files)
            example_files = [
                "quarterly_report.pdf",
                "financial_summary.pdf",
                "market_analysis.pdf"
            ]

            print("\n[SETUP] Example batch processing setup:")
            print(f"   [DIR] Input directory: {sample_docs_dir}")
            print(f"   [FILES] Example files: {len(example_files)} documents")
            for file in example_files:
                print(f"     - {file}")

            print(f"\n[HELP] To test batch processing:")
            print(f"   1. Create the directory: {sample_docs_dir}")
            print(f"   2. Place PDF files in the directory")
            print(f"   3. Run this script again")

            return True

        # Look for PDF files in sample_docs
        pdf_files = list(sample_docs_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"[INFO] No PDF files found in {sample_docs_dir}")
            print("\n[GUIDE] Batch processing would work like this:")
            print("   1. Place PDF files in sample_docs/ directory")
            print("   2. The script would process them automatically")
            print("   3. Converted files would be saved to outputs/batch_results/")

            return True

        # Validate all files before processing
        valid_files = []
        for file_path in pdf_files:
            if validate_file_path(file_path):
                valid_files.append(file_path)
            else:
                print(f"[WARN] Skipping invalid file: {file_path}")

        if not valid_files:
            print("[ERROR] No valid PDF files found for batch processing")
            return False

        print(f"[FOUND] Found {len(valid_files)} valid PDF files for batch conversion")

        # Process batch
        output_dir = Path("outputs") / "batch_results"
        print(f"[CONVERT] Starting batch conversion to: {output_dir}")

        results = granite.batch_convert(
            sources=[str(f) for f in valid_files],
            output_dir=str(output_dir)
        )

        # Analyze results
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful

        print(f"[SUCCESS] Batch processing completed!")
        print(f"   [STATS] Successfully converted: {successful}/{len(results)} documents")
        if failed > 0:
            print(f"   [ERROR] Failed conversions: {failed}")
            print("\n[ERROR] Failed files:")
            for result in results:
                if 'error' in result:
                    print(f"     - {result.get('source', 'Unknown')}: {result.get('error', 'Unknown error')}")

        return successful > 0

    except Exception as e:
        logger.error(f"Batch processing demonstration failed: {e}")
        print(f"[ERROR] Batch processing failed: {e}")
        return False


def check_models_availability() -> bool:
    """
    Check if required models are available and attempt to download if needed.

    Returns:
        bool: True if models are available, False otherwise
    """
    print("\n[CHECK] Checking model availability...")

    try:
        # Try to create a GraniteDocling instance to test if models are available
        test_granite = GraniteDocling()
        print("[PASS] Models are available and accessible")
        return True

    except Exception as e:
        logger.warning(f"Model availability check failed: {e}")
        print(f"[WARN] Model availability check failed: {e}")

        print("\n[DOWNLOAD] Attempting to download models...")
        try:
            download_models()
            print("[SUCCESS] Models downloaded successfully")

            # Test again after download
            test_granite = GraniteDocling()
            print("[PASS] Models verified after download")
            return True

        except Exception as download_error:
            logger.error(f"Model download failed: {download_error}")
            print(f"[ERROR] Model download failed: {download_error}")
            print("\n[MANUAL] Manual download instructions:")
            print("  Run the following command:")
            print("  docling-tools models download-hf-repo ibm-granite/granite-docling-258M")
            return False


def main() -> None:
    """Main example function with comprehensive error handling."""
    print("Granite Docling 258M - Basic Usage Example")
    print("=" * 60)

    try:
        # Step 1: Setup directories
        print("\n[SETUP] Setting up directories...")
        setup_directories()

        # Step 2: Check model availability
        if not check_models_availability():
            print("\n[ERROR] Cannot proceed without required models")
            print("Please follow the manual download instructions above.")
            return

        # Step 3: Initialize Granite Docling
        print("\n[INIT] Initializing Granite Docling...")
        try:
            granite = GraniteDocling()
            print("[SUCCESS] Successfully initialized Granite Docling")
        except Exception as init_error:
            logger.error(f"Failed to initialize Granite Docling: {init_error}")
            print(f"[ERROR] Initialization failed: {init_error}")
            print("\nTroubleshooting:")
            print("  - Check that all dependencies are installed: pip install -r requirements.txt")
            print("  - Verify Python version compatibility")
            print("  - Check system resources (memory, disk space)")
            return

        # Step 4: Run examples
        examples_passed = 0
        total_examples = 2

        # Example 1: Single document conversion
        if demonstrate_single_document_conversion(granite):
            examples_passed += 1

        # Example 2: Batch processing
        if demonstrate_batch_processing(granite):
            examples_passed += 1

        # Final summary
        print("\n" + "=" * 60)
        print("[COMPLETE] Basic Usage Example Completed!")
        print("=" * 60)
        print(f"[RESULTS] Examples passed: {examples_passed}/{total_examples}")

        if examples_passed == total_examples:
            print("[SUCCESS] All examples completed successfully!")
            print("\n[NEXT] Next steps:")
            print("  - Try the advanced_features.py example")
            print("  - Experiment with your own PDF documents")
            print("  - Explore batch processing with multiple files")
        elif examples_passed > 0:
            print("[WARN] Some examples completed with issues")
            print("  - Check the error messages above for details")
            print("  - Ensure network connectivity for URL-based examples")
        else:
            print("[ERROR] No examples completed successfully")
            print("  - Check system requirements and dependencies")
            print("  - Review error messages for specific issues")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Example interrupted by user")
        logger.info("Example interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"\n[ERROR] Unexpected error: {e}")
        print("\nThis is likely a system or configuration issue.")
        print("Please check:")
        print("  - Python environment and dependencies")
        print("  - Available system memory")
        print("  - Disk space for output files")
        print("  - Network connectivity")


if __name__ == "__main__":
    main()