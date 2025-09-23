#!/usr/bin/env python3
"""
Quick test of the Granite Docling implementation.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from granite_docling import GraniteDocling


def test_initialization():
    """Test basic initialization of GraniteDocling."""
    print("Testing Granite Docling initialization...")

    try:
        # Test with transformers (default)
        granite_transformers = GraniteDocling()
        print(f"  [OK] Transformers model initialized: {granite_transformers.model_type}")

        # Test with MLX
        granite_mlx = GraniteDocling(model_type="mlx")
        print(f"  [OK] MLX model initialized: {granite_mlx.model_type}")

        return True
    except Exception as e:
        print(f"  [FAIL] Initialization failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without actual conversion."""
    print("\nTesting basic functionality...")

    try:
        granite = GraniteDocling()

        # Check if converter is initialized
        if hasattr(granite, 'converter') and granite.converter is not None:
            print("  [OK] Document converter initialized")
        else:
            print("  [FAIL] Document converter not initialized")
            return False

        # Check model configuration
        if hasattr(granite, 'vlm_model'):
            print(f"  [OK] VLM model configured: {type(granite.vlm_model).__name__}")
        else:
            print("  [FAIL] VLM model not configured")
            return False

        return True
    except Exception as e:
        print(f"  [FAIL] Basic functionality test failed: {e}")
        return False


def main():
    """Run tests."""
    print("Granite Docling 258M - Quick Test")
    print("=" * 40)

    tests = [
        ("Initialization Test", test_initialization),
        ("Basic Functionality Test", test_basic_functionality),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  [ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] Granite Docling implementation is working!")
        print("\nTo test with actual documents, you need to:")
        print("1. Download models: docling-tools models download")
        print("2. Try converting a document: granite.convert_document('path/to/document.pdf')")
    else:
        print(f"\n[WARNING] {total - passed} tests failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)