#!/usr/bin/env python3
"""
Granite Docling 258M Web Interface

Interactive web interface for IBM's Granite Docling model with automatic
device detection (CPU/GPU) and optimized performance.
"""

import os
import sys
import tempfile
import json
import traceback
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import gradio as gr

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from granite_docling_gpu import GraniteDoclingGPU, DeviceManager
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    IMPORT_ERROR = str(e)


class GraniteDoclingInterface:
    """Web interface for Granite Docling with automatic device detection."""

    def __init__(self):
        """Initialize the interface with device detection."""
        self.granite_instances = {}
        self.device_manager = None

        if DOCLING_AVAILABLE:
            try:
                self.device_manager = DeviceManager()
                self.device_info = self.device_manager.get_device_info()
                self.available_devices = self.device_manager.detect_available_devices()
            except Exception as e:
                print(f"Warning: Could not initialize device manager: {e}")

    def get_optimal_device_recommendation(self) -> str:
        """Get a recommendation for the optimal device."""
        if not DOCLING_AVAILABLE or not self.device_manager:
            return "CPU (Default)"

        optimal = self.device_manager.get_optimal_device()
        available = self.available_devices

        if optimal == "CUDA" and "CUDA" in available:
            return "🚀 CUDA GPU (Recommended - Fastest)"
        elif optimal == "MPS" and "MPS" in available:
            return "🍎 Apple MPS (Recommended - Apple Silicon)"
        else:
            return "💻 CPU (Recommended - Universal)"

    def get_or_create_granite_instance(self, model_type: str, device: str) -> Optional['GraniteDoclingGPU']:
        """Get or create a Granite instance for the specified configuration."""
        key = f"{model_type}_{device}"

        if key not in self.granite_instances:
            try:
                if device.lower() == "auto":
                    granite = GraniteDoclingGPU(model_type=model_type, auto_device=True)
                else:
                    granite = GraniteDoclingGPU(model_type=model_type, device=device, auto_device=False)
                self.granite_instances[key] = granite
            except Exception as e:
                print(f"Failed to create Granite instance: {e}")
                return None

        return self.granite_instances[key]

    def process_document_with_gpu(
        self,
        file_input,
        processing_mode: str,
        model_type: str,
        device: str,
        include_metadata: bool = True,
        show_device_info: bool = True
    ) -> Tuple[str, str, str, str, str]:
        """
        Process uploaded document with GPU support and selected options.

        Returns: (markdown_output, json_metadata, processing_info, device_status, error_message)
        """
        if not DOCLING_AVAILABLE:
            error_msg = f"Docling not available: {IMPORT_ERROR}"
            return "", "", "", "", error_msg

        if file_input is None:
            return "", "", "", "", "Please upload a file first."

        try:
            # Get or create the appropriate Granite instance
            granite = self.get_or_create_granite_instance(model_type, device)
            if granite is None:
                return "", "", "", "", "Failed to initialize Granite Docling model."

            # Start timing
            start_time = time.time()

            # Processing info
            actual_device = granite.device
            processing_info = f"🔧 Processing with {model_type} backend on {actual_device}...\n"

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
                    result = granite.convert_document(temp_file, show_device_info=show_device_info)
                    markdown_output = result["content"]

                elif processing_mode == "Document Analysis":
                    result = granite.convert_document(temp_file, show_device_info=show_device_info)
                    # Analyze the document structure
                    lines = result["content"].split('\n')
                    headers = [line for line in lines if line.startswith('#')]
                    tables = [line for line in lines if '|' in line and line.strip()]

                    analysis = f"""# Document Analysis (GPU-Accelerated)

## Processing Information
- **Device Used**: {actual_device}
- **Model Type**: {model_type}
- **Processing Time**: {time.time() - start_time:.2f}s

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
                    result = granite.convert_document(temp_file, show_device_info=show_device_info)
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
                        markdown_output = f"""# Extracted Tables (GPU-Accelerated)

**Device**: {actual_device} | **Processing Time**: {time.time() - start_time:.2f}s

{chr(10).join(table_lines)}"""
                    else:
                        markdown_output = f"""# No Tables Found

**Device**: {actual_device} | **Processing Time**: {time.time() - start_time:.2f}s

No table structures were detected in this document."""

                elif processing_mode == "Quick Preview":
                    result = granite.convert_document(temp_file, show_device_info=show_device_info)
                    # Show first 1000 characters
                    preview = result["content"][:1000]
                    if len(result["content"]) > 1000:
                        preview += "\n\n... (truncated)"
                    markdown_output = f"""# Quick Preview (GPU-Accelerated)

**Device**: {actual_device} | **Processing Time**: {time.time() - start_time:.2f}s

{preview}"""
                else:
                    result = granite.convert_document(temp_file, show_device_info=show_device_info)
                    markdown_output = result["content"]

                # Calculate processing time
                processing_time = time.time() - start_time

                # Prepare metadata
                metadata = {
                    "processing_mode": processing_mode,
                    "model_type": model_type,
                    "device_requested": device,
                    "device_actual": actual_device,
                    "file_name": getattr(file_input, 'name', 'uploaded_file'),
                    "content_length": len(markdown_output),
                    "processing_time_seconds": round(processing_time, 2),
                    "processing_successful": True
                }

                if 'metadata' in result:
                    metadata.update(result['metadata'])

                json_metadata = json.dumps(metadata, indent=2) if include_metadata else ""

                # Get device status
                device_status = ""
                if show_device_info and granite:
                    try:
                        status = granite.get_device_status()
                        device_status = json.dumps(status, indent=2)
                    except Exception as e:
                        device_status = f"Device status error: {str(e)}"

                processing_info = f"""✅ Successfully processed with {model_type} backend
🖥️  Device: {actual_device}
⚡ Mode: {processing_mode}
⏱️  Processing time: {processing_time:.2f}s
📄 Content length: {len(markdown_output)} characters"""

                return markdown_output, json_metadata, processing_info, device_status, ""

            finally:
                # Clean up temp file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        except Exception as e:
            error_msg = f"Error processing document: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return "", "", "", "", error_msg

    def create_interface(self) -> gr.Interface:
        """Create the enhanced Gradio interface with GPU support."""

        # Custom CSS for better styling with dark theme compatibility
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            color: #4a90e2;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: rgba(74, 144, 226, 0.1);
            color: var(--body-text-color, #ffffff);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid rgba(74, 144, 226, 0.3);
        }
        .gpu-box {
            background-color: rgba(76, 175, 80, 0.1);
            color: var(--body-text-color, #ffffff);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }
        .error-box {
            background-color: rgba(244, 67, 54, 0.1);
            color: var(--body-text-color, #ffffff);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid rgba(244, 67, 54, 0.3);
        }
        .device-status {
            background-color: rgba(255, 152, 0, 0.1);
            color: var(--body-text-color, #ffffff);
            padding: 10px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            border: 1px solid rgba(255, 152, 0, 0.3);
        }
        /* Dark theme specific styles */
        .dark .info-box, .dark .gpu-box, .dark .error-box, .dark .device-status {
            color: #ffffff;
        }
        /* Ensure readability in both light and dark modes */
        .info-box h3, .info-box h4, .gpu-box h3, .gpu-box h4, .error-box h3, .error-box h4 {
            color: inherit;
            margin-top: 0;
        }
        .info-box ul, .gpu-box ul, .error-box ul {
            color: inherit;
        }
        .info-box strong, .gpu-box strong, .error-box strong {
            color: inherit;
            font-weight: 600;
        }
        """

        with gr.Blocks(css=css, title="Granite Docling GPU Demo") as interface:

            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1>🚀 Granite Docling 258M - GPU Accelerated Demo</h1>
                    <p>Upload documents and experience high-performance AI document processing with GPU acceleration</p>
                </div>
            """)

            # GPU Status Section
            if DOCLING_AVAILABLE and self.device_manager:
                device_recommendation = self.get_optimal_device_recommendation()
                gpu_info = f"""
                    <div class="gpu-box">
                        <h3>🖥️ Hardware Status</h3>
                        <p><strong>Available Devices:</strong> {', '.join(self.available_devices)}</p>
                        <p><strong>Recommendation:</strong> {device_recommendation}</p>
                        <p><em>GPU acceleration significantly improves processing speed for large documents</em></p>
                    </div>
                """
                gr.HTML(gpu_info)

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

                    # GPU/Device selection
                    device_choices = ["Auto"] + (self.available_devices if DOCLING_AVAILABLE else ["CPU"])
                    device_selection = gr.Dropdown(
                        choices=device_choices,
                        label="🖥️ Processing Device",
                        value="Auto",
                        info="Select processing device (Auto = best available)"
                    )

                    model_type = gr.Dropdown(
                        choices=["transformers", "mlx"],
                        label="Model Backend",
                        value="transformers",
                        info="Transformers (universal) or MLX (Apple Silicon)"
                    )

                    with gr.Row():
                        include_metadata = gr.Checkbox(
                            label="Include Processing Metadata",
                            value=True
                        )
                        show_device_info = gr.Checkbox(
                            label="Show Device Performance Info",
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
                        lines=6,
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
                                lines=12
                            )

                        with gr.TabItem("🖥️ Device Status"):
                            device_status = gr.Code(
                                label="Device Performance Info",
                                language="json",
                                lines=10
                            )

                        with gr.TabItem("❌ Errors"):
                            error_output = gr.Textbox(
                                label="Error Messages",
                                lines=8,
                                interactive=False
                            )

            # Performance and Tips section
            gr.HTML("<h3>⚡ Performance & GPU Tips</h3>")

            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                        <div class="info-box">
                            <h4>🚀 GPU Acceleration Benefits:</h4>
                            <ul>
                                <li><strong>CUDA GPU:</strong> 3-5x faster processing for large documents</li>
                                <li><strong>Apple MPS:</strong> 2-3x faster on Apple Silicon Macs</li>
                                <li><strong>Batch Processing:</strong> Even greater speedups for multiple documents</li>
                                <li><strong>Memory Efficiency:</strong> Better handling of large files</li>
                            </ul>
                        </div>
                    """)

                with gr.Column():
                    gr.HTML("""
                        <div class="info-box">
                            <h4>💡 Performance Tips:</h4>
                            <ul>
                                <li>Use "Auto" device selection for optimal performance</li>
                                <li>GPU acceleration is most beneficial for PDFs with images</li>
                                <li>MLX backend is optimized for Apple Silicon</li>
                                <li>Monitor device status to track GPU memory usage</li>
                            </ul>
                        </div>
                    """)

            # Event handlers
            process_btn.click(
                fn=self.process_document_with_gpu,
                inputs=[file_input, processing_mode, model_type, device_selection, include_metadata, show_device_info],
                outputs=[markdown_output, json_output, processing_info, device_status, error_output]
            )

            # Hardware requirements
            gr.HTML("""
                <div class="info-box">
                    <h4>🖥️ Hardware Requirements:</h4>
                    <ul>
                        <li><strong>CPU:</strong> Any modern processor (universal compatibility)</li>
                        <li><strong>NVIDIA GPU:</strong> CUDA-compatible GPU with 4GB+ VRAM (recommended)</li>
                        <li><strong>Apple Silicon:</strong> M1/M2/M3 Macs with 8GB+ unified memory</li>
                        <li><strong>RAM:</strong> 8GB minimum, 16GB+ recommended for large documents</li>
                    </ul>
                </div>
            """)

        return interface

    def launch(self, **kwargs):
        """Launch the enhanced Gradio interface."""
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
            if self.device_manager:
                optimal_device = self.device_manager.get_optimal_device()
                print(f"🖥️  Optimal device detected: {optimal_device}")
                available = self.available_devices
                print(f"🔧 Available devices: {', '.join(available)}")
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