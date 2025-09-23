#!/usr/bin/env python3
"""
Gradio Web Interface for Granite Docling 258M

This creates an interactive web interface where users can upload documents
and see different processing capabilities of the Granite Docling model.
"""

import os
import sys
import tempfile
import json
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import gradio as gr
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from granite_docling import GraniteDocling
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    IMPORT_ERROR = str(e)


class GraniteDoclingInterface:
    """Web interface for Granite Docling demonstration."""

    def __init__(self):
        """Initialize the interface."""
        self.granite_transformers = None
        self.granite_mlx = None

        if DOCLING_AVAILABLE:
            try:
                # Initialize both model types
                self.granite_transformers = GraniteDocling(model_type="transformers")
                # MLX might not be available on all systems
                try:
                    self.granite_mlx = GraniteDocling(model_type="mlx")
                except Exception:
                    self.granite_mlx = None
            except Exception as e:
                print(f"Warning: Could not initialize Granite Docling: {e}")

    def process_document(
        self,
        file_input,
        processing_mode: str,
        model_type: str,
        include_metadata: bool = True
    ) -> Tuple[str, str, str, str]:
        """
        Process uploaded document with selected options.

        Returns: (markdown_output, json_metadata, processing_info, error_message)
        """
        if not DOCLING_AVAILABLE:
            error_msg = f"Docling not available: {IMPORT_ERROR}"
            return "", "", "", error_msg

        if file_input is None:
            return "", "", "", "Please upload a file first."

        try:
            # Select the appropriate model
            granite = self.granite_transformers
            if model_type == "mlx" and self.granite_mlx is not None:
                granite = self.granite_mlx
            elif model_type == "mlx":
                return "", "", "", "MLX model not available on this system."

            if granite is None:
                return "", "", "", "Granite Docling model not initialized."

            # Process the document
            processing_info = f"Processing with {model_type} backend..."

            # Save uploaded file to temporary location
            temp_file = None
            try:
                # Create temp file with original extension
                file_ext = Path(file_input.name).suffix if hasattr(file_input, 'name') else '.tmp'
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    if hasattr(file_input, 'read'):
                        tmp.write(file_input.read())
                    else:
                        # Handle file path case
                        with open(file_input, 'rb') as f:
                            tmp.write(f.read())
                    temp_file = tmp.name

                # Process based on selected mode
                if processing_mode == "Full Markdown Conversion":
                    result = granite.convert_document(temp_file)
                    markdown_output = result["content"]

                elif processing_mode == "Document Analysis":
                    result = granite.convert_document(temp_file)
                    # Analyze the document structure
                    lines = result["content"].split('\n')
                    headers = [line for line in lines if line.startswith('#')]
                    tables = [line for line in lines if '|' in line and line.strip()]

                    analysis = f"""# Document Analysis

## Structure Overview
- Total lines: {len(lines)}
- Headers found: {len(headers)}
- Potential tables: {len([line for line in tables if line.count('|') > 1])}
- Pages processed: {result.get('pages', 1)}

## Headers Found:
{chr(10).join(headers[:10]) if headers else "No headers detected"}

## Sample Content:
{chr(10).join(lines[:20])}
"""
                    markdown_output = analysis

                elif processing_mode == "Table Extraction":
                    result = granite.convert_document(temp_file)
                    # Extract table-like content
                    lines = result["content"].split('\n')
                    table_lines = []
                    in_table = False

                    for line in lines:
                        if '|' in line and line.strip():
                            table_lines.append(line)
                            in_table = True
                        elif in_table and not line.strip():
                            table_lines.append("")
                        elif in_table and line.strip() and '|' not in line:
                            in_table = False

                    if table_lines:
                        markdown_output = f"# Extracted Tables\n\n{chr(10).join(table_lines)}"
                    else:
                        markdown_output = "# No Tables Found\n\nNo table structures were detected in this document."

                elif processing_mode == "Quick Preview":
                    result = granite.convert_document(temp_file)
                    # Show first 1000 characters
                    preview = result["content"][:1000]
                    if len(result["content"]) > 1000:
                        preview += "\n\n... (truncated)"
                    markdown_output = f"# Quick Preview\n\n{preview}"
                else:
                    result = granite.convert_document(temp_file)
                    markdown_output = result["content"]

                # Prepare metadata
                metadata = {
                    "processing_mode": processing_mode,
                    "model_type": model_type,
                    "file_name": getattr(file_input, 'name', 'uploaded_file'),
                    "content_length": len(markdown_output),
                    "processing_successful": True
                }

                if 'metadata' in result:
                    metadata.update(result['metadata'])

                json_metadata = json.dumps(metadata, indent=2) if include_metadata else ""

                processing_info = f"✅ Successfully processed with {model_type} backend\n" \
                                f"Mode: {processing_mode}\n" \
                                f"Content length: {len(markdown_output)} characters"

                return markdown_output, json_metadata, processing_info, ""

            finally:
                # Clean up temp file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        except Exception as e:
            error_msg = f"Error processing document: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return "", "", "", error_msg

    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""

        # Custom CSS for better styling
        css = """
        .gradio-container {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .main-header {text-align: center; color: #2d5aa0; margin-bottom: 20px;}
        .info-box {background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin: 10px 0;}
        .error-box {background-color: #ffe6e6; padding: 15px; border-radius: 8px; margin: 10px 0;}
        .success-box {background-color: #e6ffe6; padding: 15px; border-radius: 8px; margin: 10px 0;}
        """

        with gr.Blocks(css=css, title="Granite Docling 258M Demo") as interface:

            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1>🔬 Granite Docling 258M Interactive Demo</h1>
                    <p>Upload documents and explore different processing capabilities of IBM's Granite Docling model</p>
                </div>
            """)

            # Status check
            if not DOCLING_AVAILABLE:
                gr.HTML(f"""
                    <div class="error-box">
                        <h3>⚠️ Setup Required</h3>
                        <p>Docling is not available: {IMPORT_ERROR}</p>
                        <p>Please run: <code>pip install -r requirements.txt</code></p>
                    </div>
                """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.HTML("<h3>📤 Upload & Configure</h3>")

                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"],
                        type="filepath"
                    )

                    processing_mode = gr.Dropdown(
                        choices=[
                            "Full Markdown Conversion",
                            "Document Analysis",
                            "Table Extraction",
                            "Quick Preview"
                        ],
                        label="Processing Mode",
                        value="Full Markdown Conversion",
                        info="Choose what type of processing to perform"
                    )

                    model_type = gr.Dropdown(
                        choices=["transformers", "mlx"],
                        label="Model Backend",
                        value="transformers",
                        info="Transformers (universal) or MLX (Apple Silicon)"
                    )

                    include_metadata = gr.Checkbox(
                        label="Include Processing Metadata",
                        value=True
                    )

                    process_btn = gr.Button(
                        "🚀 Process Document",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    # Output section
                    gr.HTML("<h3>📊 Results</h3>")

                    # Processing status
                    processing_info = gr.Textbox(
                        label="Processing Status",
                        lines=3,
                        interactive=False
                    )

                    # Main output tabs
                    with gr.Tabs():
                        with gr.TabItem("📝 Processed Content"):
                            markdown_output = gr.Markdown(
                                label="Markdown Output",
                                height=400
                            )

                        with gr.TabItem("🔧 Metadata"):
                            json_output = gr.Code(
                                label="Processing Metadata",
                                language="json",
                                lines=10
                            )

                        with gr.TabItem("❌ Errors"):
                            error_output = gr.Textbox(
                                label="Error Messages",
                                lines=8,
                                interactive=False
                            )

            # Examples section
            gr.HTML("<h3>💡 Tips & Examples</h3>")

            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                        <div class="info-box">
                            <h4>Processing Modes:</h4>
                            <ul>
                                <li><strong>Full Markdown:</strong> Complete document conversion</li>
                                <li><strong>Document Analysis:</strong> Structure and content analysis</li>
                                <li><strong>Table Extraction:</strong> Focus on extracting tables</li>
                                <li><strong>Quick Preview:</strong> Fast preview of first 1000 characters</li>
                            </ul>
                        </div>
                    """)

                with gr.Column():
                    gr.HTML("""
                        <div class="info-box">
                            <h4>Supported Formats:</h4>
                            <ul>
                                <li>📄 PDF documents</li>
                                <li>📝 Word documents (.docx, .doc)</li>
                                <li>🖼️ Images (.png, .jpg, .jpeg)</li>
                            </ul>
                            <p><em>Note: Processing time varies by document size and complexity</em></p>
                        </div>
                    """)

            # Event handlers
            process_btn.click(
                fn=self.process_document,
                inputs=[file_input, processing_mode, model_type, include_metadata],
                outputs=[markdown_output, json_output, processing_info, error_output]
            )

            # Example files (if we had them)
            gr.HTML("""
                <div class="info-box">
                    <h4>🎯 Getting Started:</h4>
                    <ol>
                        <li>Upload a PDF, Word document, or image</li>
                        <li>Choose your preferred processing mode</li>
                        <li>Select model backend (use 'transformers' for most systems)</li>
                        <li>Click 'Process Document' and wait for results</li>
                    </ol>
                </div>
            """)

        return interface

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()

        # Default launch parameters
        launch_params = {
            "server_name": "127.0.0.1",
            "server_port": 7860,
            "share": False,
            "debug": False,
            "show_error": True
        }
        launch_params.update(kwargs)

        print("🚀 Starting Granite Docling Interactive Demo...")
        print(f"📍 Interface will be available at: http://{launch_params['server_name']}:{launch_params['server_port']}")

        if DOCLING_AVAILABLE:
            print("✅ Granite Docling models loaded and ready")
        else:
            print("⚠️  Docling not available - interface will show setup instructions")

        interface.launch(**launch_params)


def main():
    """Main function to run the interface."""
    print("Granite Docling 258M - Interactive Web Demo")
    print("=" * 50)

    # Create and launch interface
    demo = GraniteDoclingInterface()
    demo.launch(
        share=False,  # Set to True to create public link
        debug=True,   # Enable debug mode
        server_name="127.0.0.1",
        server_port=7860
    )


if __name__ == "__main__":
    main()