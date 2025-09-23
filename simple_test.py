#!/usr/bin/env python3
"""
Simple test script to verify the Granite Docling implementation setup.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")

    try:
        import docling
        print("  [OK] docling imported successfully")
    except ImportError as e:
        print(f"  [FAIL] docling import failed: {e}")
        return False

    try:
        from docling.document_converter import DocumentConverter
        print("  [OK] DocumentConverter imported successfully")
    except ImportError as e:
        print(f"  [FAIL] DocumentConverter import failed: {e}")
        return False

    try:
        from granite_docling import GraniteDocling
        print("  [OK] GraniteDocling imported successfully")
    except ImportError as e:
        print(f"  [FAIL] GraniteDocling import failed: {e}")
        return False

    return True


def test_directory_structure():
    """Test if all required directories and files exist."""
    print("\nTesting directory structure...")

    required_files = [
        "src/granite_docling.py",
        "examples/basic_usage.py",
        "examples/advanced_features.py",
        "requirements.txt",
        "README.md"
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  [OK] {file_path} exists")
        else:
            print(f"  [FAIL] {file_path} missing")
            all_exist = False

    return all_exist


def main():
    """Run basic tests."""
    print("Granite Docling 258M - Simple Setup Test")
    print("=" * 45)

    # Test 1: Directory structure
    dir_test = test_directory_structure()

    # Test 2: Basic imports (only if docling is installed)
    import_test = False
    try:
        import_test = test_imports()
    except Exception as e:
        print(f"\nImport test skipped - dependencies not installed: {e}")
        print("Run 'pip install -r requirements.txt' to install dependencies")

    print("\nTest Summary:")
    print(f"Directory structure: {'PASS' if dir_test else 'FAIL'}")
    print(f"Module imports: {'PASS' if import_test else 'SKIP/FAIL'}")

    if dir_test:
        print("\n[OK] Project structure is correct!")
        if not import_test:
            print("\nNext steps:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Download models: docling-tools models download-hf-repo ibm-granite/granite-docling-258M")
            print("3. Run examples: python examples/basic_usage.py")
    else:
        print("\n[FAIL] Project structure has issues. Please check missing files.")

    return dir_test


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)