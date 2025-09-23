#!/usr/bin/env python3
"""
Simple setup check for Granite Docling demo.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Check setup status."""
    print("Granite Docling Setup Check")
    print("=" * 30)

    # Check Python version
    print(f"Python: {sys.version.split()[0]}")

    # Check dependencies
    try:
        import gradio
        print(f"[OK] Gradio: {gradio.__version__}")
    except ImportError:
        print("[FAIL] Gradio not installed")

    try:
        import docling
        print("[OK] Docling: Available")
    except ImportError as e:
        print(f"[FAIL] Docling: {e}")

    try:
        from granite_docling import GraniteDocling
        granite = GraniteDocling()
        print(f"[OK] Granite Docling: {granite.model_type}")
    except Exception as e:
        print(f"[FAIL] Granite Docling: {e}")

    # Check models
    home_dir = Path.home()
    cache_dir = home_dir / ".cache" / "docling"
    if cache_dir.exists():
        print(f"[OK] Cache directory: {cache_dir}")
    else:
        print("[INFO] No model cache found")
        print("       Run: docling-tools models download")

    print("\nReady to launch demo!")
    print("Run: python simple_demo.py")


if __name__ == "__main__":
    main()