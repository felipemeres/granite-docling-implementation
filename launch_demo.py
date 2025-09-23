#!/usr/bin/env python3
"""
Launch script for Granite Docling demo interfaces.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def launch_simple_demo():
    """Launch the simple demo."""
    print("🚀 Launching Simple Demo...")
    print("This will start a basic interface for testing setup.")
    print("Open: http://127.0.0.1:7860")
    print("-" * 50)

    try:
        from simple_demo import create_demo
        demo = create_demo()
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")


def launch_full_demo():
    """Launch the full-featured demo."""
    print("🚀 Launching Full Demo...")
    print("This will start the complete interface with all features.")
    print("Open: http://127.0.0.1:7861")
    print("-" * 50)

    try:
        from gradio_interface import GraniteDoclingInterface
        demo = GraniteDoclingInterface()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")


def check_setup():
    """Check if everything is set up correctly."""
    print("🔍 Checking Setup...")
    print("-" * 30)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check if dependencies are available
    try:
        import gradio
        print(f"✅ Gradio: {gradio.__version__}")
    except ImportError:
        print("❌ Gradio not installed")

    try:
        import docling
        print("✅ Docling: Available")
    except ImportError as e:
        print(f"❌ Docling not available: {e}")

    try:
        from granite_docling import GraniteDocling
        granite = GraniteDocling()
        print("✅ Granite Docling: Ready")
    except Exception as e:
        print(f"❌ Granite Docling: {e}")

    # Check for model files
    print("\n📁 Model Status:")
    home_dir = Path.home()
    cache_dir = home_dir / ".cache" / "docling" / "models"
    if cache_dir.exists():
        print(f"✅ Model cache found: {cache_dir}")
        model_files = list(cache_dir.rglob("*"))
        print(f"   Models found: {len(model_files)} files")
    else:
        print("❌ Model cache not found")
        print("   Run: docling-tools models download")


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Granite Docling Demo Launcher")
    parser.add_argument(
        "mode",
        choices=["simple", "full", "check"],
        help="Demo mode to launch"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )

    args = parser.parse_args()

    print("Granite Docling 258M - Demo Launcher")
    print("=" * 45)

    if args.mode == "check":
        check_setup()
    elif args.mode == "simple":
        launch_simple_demo()
    elif args.mode == "full":
        launch_full_demo()


if __name__ == "__main__":
    main()