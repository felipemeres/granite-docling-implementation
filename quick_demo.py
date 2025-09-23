#!/usr/bin/env python3
"""
Quick demonstration of Granite Docling 258M usage
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from granite_docling import GraniteDocling


def main():
    print("Granite Docling 258M - Quick Demo")
    print("=" * 40)

    # Initialize the processor
    print("1. Initializing Granite Docling...")
    granite = GraniteDocling(model_type="transformers")
    print("   [OK] Ready!")

    # Show available model types
    print("\n2. Available model types:")
    print("   - 'transformers' (default) - Uses Hugging Face Transformers")
    print("   - 'mlx' - Uses Apple MLX framework (for Apple Silicon Macs)")

    # Example usage patterns
    print("\n3. Usage examples:")

    print("\n   Basic document conversion:")
    print("   ```python")
    print("   from granite_docling import GraniteDocling")
    print("   ")
    print("   granite = GraniteDocling()")
    print("   result = granite.convert_document('document.pdf')")
    print("   print(result['content'])  # Markdown output")
    print("   ```")

    print("\n   Convert and save to file:")
    print("   ```python")
    print("   granite.convert_to_file('document.pdf', 'output.md')")
    print("   ```")

    print("\n   Batch conversion:")
    print("   ```python")
    print("   files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']")
    print("   results = granite.batch_convert(files, 'output_directory/')")
    print("   ```")

    print("\n   Using MLX (Apple Silicon):")
    print("   ```python")
    print("   granite_mlx = GraniteDocling(model_type='mlx')")
    print("   result = granite_mlx.convert_document('document.pdf')")
    print("   ```")

    print("\n4. Prerequisites for actual document conversion:")
    print("   - Download models: `docling-tools models download`")
    print("   - Have documents in supported formats (PDF, DOCX, etc.)")
    print("   - Ensure sufficient system resources for model inference")

    print("\n5. Key features:")
    print("   [+] Vision-Language Model for document understanding")
    print("   [+] Supports multiple document formats")
    print("   [+] Outputs structured Markdown")
    print("   [+] Batch processing capabilities")
    print("   [+] Configurable model backends (Transformers/MLX)")
    print("   [+] Built on IBM's Granite foundation models")

    print(f"\n6. Model Information:")
    print(f"   Current model type: {granite.model_type}")
    print(f"   Model configuration: {type(granite.vlm_model).__name__}")

    print("\n" + "=" * 40)
    print("Demo completed! [SUCCESS]")
    print("\nNext steps:")
    print("- Try the examples in the examples/ directory")
    print("- Run 'python examples/basic_usage.py' for a full example")
    print("- Check the README.md for more detailed information")


if __name__ == "__main__":
    main()