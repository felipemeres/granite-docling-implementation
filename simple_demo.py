#!/usr/bin/env python3
"""
Simple Gradio Demo for Granite Docling 258M

A minimal interface to test the setup and basic functionality.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from granite_docling import GraniteDocling
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    IMPORT_ERROR = str(e)


def process_simple(file_input, model_type):
    """Simple processing function for testing."""
    if not DOCLING_AVAILABLE:
        return f"❌ Docling not available: {IMPORT_ERROR}", "Setup required"

    if file_input is None:
        return "📤 Please upload a file first.", "No file"

    try:
        # Initialize the model
        granite = GraniteDocling(model_type=model_type)

        # For testing, just return basic info without actual processing
        file_info = f"""# File Information

**Filename:** {Path(file_input).name if isinstance(file_input, str) else 'uploaded_file'}
**Model Type:** {model_type}
**Status:** ✅ Ready for processing

## Next Steps
To enable full document processing:
1. Run: `docling-tools models download`
2. Upload a PDF or image file
3. The model will convert it to structured Markdown

## Model Info
- **Type:** {granite.model_type}
- **Configuration:** {type(granite.vlm_model).__name__}
- **Status:** Initialized successfully
"""

        return file_info, "✅ Ready"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "Error"


# Create simple interface
def create_demo():
    """Create a simple demo interface."""

    with gr.Blocks(title="Granite Docling Demo") as demo:
        gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🧪 Granite Docling 258M - Simple Demo</h1>
                <p>Test the setup and basic functionality</p>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Document (PDF, Image, etc.)",
                    file_types=[".pdf", ".png", ".jpg", ".docx"]
                )

                model_type = gr.Radio(
                    choices=["transformers", "mlx"],
                    label="Model Type",
                    value="transformers"
                )

                process_btn = gr.Button("🚀 Test Setup", variant="primary")

            with gr.Column():
                output_text = gr.Markdown(label="Output")
                status_text = gr.Textbox(label="Status", lines=2)

        process_btn.click(
            fn=process_simple,
            inputs=[file_input, model_type],
            outputs=[output_text, status_text]
        )

        # Setup instructions
        gr.HTML("""
            <div style="background: #f0f8ff; padding: 15px; margin: 20px; border-radius: 8px;">
                <h3>🔧 Setup Instructions</h3>
                <ol>
                    <li>Make sure dependencies are installed: <code>pip install -r requirements.txt</code></li>
                    <li>Download models: <code>docling-tools models download</code></li>
                    <li>Upload a test file and click "Test Setup"</li>
                </ol>
            </div>
        """)

    return demo


if __name__ == "__main__":
    print("Starting Simple Granite Docling Demo...")
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True
    )