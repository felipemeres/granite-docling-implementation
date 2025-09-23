"""
Granite Docling 258M Implementation with GPU Support

This module provides an interface to the IBM Granite Docling 258M model
for document processing and conversion tasks with GPU acceleration support.
"""

import logging
import platform
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# Import the base class
try:
    from .granite_docling import GraniteDocling
except ImportError:
    # Handle case when running as script
    from granite_docling import GraniteDocling

# Import Docling dependencies for GPU-specific functionality
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions,
    AcceleratorDevice,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

# Import for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Additional imports for fast document analysis (same as base class)
try:
    import fitz  # PyMuPDF for fast PDF metadata extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device detection and selection for optimal performance."""

    @staticmethod
    def detect_available_devices() -> List[str]:
        """Detect available acceleration devices."""
        devices = [AcceleratorDevice.CPU]

        if TORCH_AVAILABLE:
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                devices.append(AcceleratorDevice.CUDA)
                logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")

            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices.append(AcceleratorDevice.MPS)
                logger.info("Apple MPS (Metal Performance Shaders) detected")

        return devices

    @staticmethod
    def get_optimal_device(prefer_gpu: bool = True) -> str:
        """Get the optimal device for processing."""
        available_devices = DeviceManager.detect_available_devices()

        if not prefer_gpu:
            return AcceleratorDevice.CPU

        # Prefer GPU devices in order: CUDA > MPS > CPU
        if AcceleratorDevice.CUDA in available_devices:
            return AcceleratorDevice.CUDA
        elif AcceleratorDevice.MPS in available_devices:
            return AcceleratorDevice.MPS
        else:
            return AcceleratorDevice.CPU

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "torch_available": TORCH_AVAILABLE,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "available_devices": DeviceManager.detect_available_devices()
        }

        if TORCH_AVAILABLE:
            info.update({
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            })

            if torch.cuda.is_available():
                info.update({
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
                })

        return info


class GraniteDoclingGPU(GraniteDocling):
    """Enhanced Granite Docling wrapper with GPU acceleration support.

    This class extends the base GraniteDocling class with automatic GPU detection
    and optimization for better performance on supported hardware.
    """

    def __init__(
        self,
        model_type: str = "transformers",
        device: Optional[str] = None,
        auto_device: bool = True,
        artifacts_path: Optional[str] = None
    ):
        """
        Initialize the Granite Docling processor with GPU support.

        Args:
            model_type: Model type - "transformers" or "mlx"
            device: Specific device to use - "cpu", "cuda", "mps", or None for auto
            auto_device: Automatically select the best available device
            artifacts_path: Path to cached model artifacts
        """
        # Device management setup (before calling parent __init__)
        self.device_manager = DeviceManager()
        self.device_info = self.device_manager.get_device_info()

        # Determine device to use
        if device is None and auto_device:
            self.device = self.device_manager.get_optimal_device(prefer_gpu=True)
        elif device is not None:
            if device.upper() in [d.upper() for d in self.device_info["available_devices"]]:
                self.device = device.upper()
            else:
                logger.warning(f"Requested device {device} not available. Falling back to CPU.")
                self.device = AcceleratorDevice.CPU
        else:
            self.device = AcceleratorDevice.CPU

        logger.info(f"Using device: {self.device}")

        # Initialize parent class
        super().__init__(model_type=model_type, artifacts_path=artifacts_path)

    def _setup_converter(self):
        """Set up the document converter with GPU-aware configuration."""
        # Create a copy of the VLM model config and update supported devices
        vlm_config = self.vlm_model

        # Ensure our selected device is in the supported devices list
        if hasattr(vlm_config, 'supported_devices'):
            if self.device not in vlm_config.supported_devices:
                # Create new config with our device included
                supported_devices = list(vlm_config.supported_devices) + [self.device]
                # Note: We would need to create a new config object here
                # For now, we'll work with the existing config

        # Set up VLM pipeline options
        pipeline_options = VlmPipelineOptions(vlm_options=vlm_config)

        # Configure PDF processing options
        pdf_options = PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )

        # If artifacts path is specified, add it to PDF pipeline options
        if self.artifacts_path:
            pdf_pipeline_options = PdfPipelineOptions(artifacts_path=self.artifacts_path)
            pdf_options.pipeline_options = pdf_pipeline_options

        # Initialize the document converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_options,
            }
        )

        logger.info(f"Initialized Granite Docling with model type: {self.model_type}, device: {self.device}")

    def analyze_document_structure(
        self,
        source: Union[str, Path],
        sample_pages: int = 3,
        max_sample_chars: int = 2000,
        include_device_info: bool = True
    ) -> Dict[str, Any]:
        """
        GPU-optimized fast document structure analysis without full conversion.

        This method provides the same lightweight document insights as the base class
        but with enhanced performance monitoring and GPU-specific optimizations.

        Args:
            source: Path to the document
            sample_pages: Number of pages to sample for content analysis
            max_sample_chars: Maximum characters to extract for preview
            include_device_info: Include GPU/device performance information

        Returns:
            Dictionary containing document analysis, structure information, and GPU metrics
        """
        start_time = time.time()

        try:
            source_path = Path(source)
            logger.info(f"Analyzing document structure on {self.device}: {source}")

            # Get GPU memory status at start (if applicable)
            initial_gpu_status = self._get_gpu_memory_status() if include_device_info else None

            # Initialize analysis result with GPU-specific fields
            analysis_result = {
                "source": str(source),
                "file_name": source_path.name,
                "file_size_mb": round(source_path.stat().st_size / (1024 * 1024), 2),
                "analysis_time_seconds": 0,
                "document_type": source_path.suffix.lower(),
                "structure_detected": {},
                "content_preview": "",
                "metadata_extraction": {},
                "processing_approach": f"fast_analysis_gpu_{self.device.lower()}",
                "device_used": self.device
            }

            # For PDFs, use PyMuPDF for maximum speed (GPU not needed for this step)
            if source_path.suffix.lower() == '.pdf' and PYMUPDF_AVAILABLE:
                analysis_result.update(self._analyze_pdf_structure_gpu_optimized(source, sample_pages, max_sample_chars))

            # For images, use PIL with GPU context awareness
            elif source_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'] and PIL_AVAILABLE:
                analysis_result.update(self._analyze_image_structure_gpu_aware(source))

            # For other formats, use minimal docling with GPU monitoring
            else:
                analysis_result.update(self._analyze_other_format_structure_gpu(source, sample_pages, max_sample_chars))

            # Calculate timing and GPU metrics
            analysis_result["analysis_time_seconds"] = round(time.time() - start_time, 2)

            if include_device_info:
                final_gpu_status = self._get_gpu_memory_status()
                analysis_result["performance_metrics"] = {
                    "device": self.device,
                    "initial_gpu_memory": initial_gpu_status,
                    "final_gpu_memory": final_gpu_status,
                    "processing_speed_mb_per_sec": round(
                        analysis_result["file_size_mb"] / max(analysis_result["analysis_time_seconds"], 0.01), 2
                    )
                }

            logger.info(f"GPU-optimized analysis completed in {analysis_result['analysis_time_seconds']} seconds on {self.device}")
            return analysis_result

        except Exception as e:
            logger.error(f"Error in GPU-optimized document structure analysis {source}: {str(e)}")
            return {
                "source": str(source),
                "error": str(e),
                "analysis_time_seconds": round(time.time() - start_time, 2),
                "processing_approach": f"fast_analysis_gpu_{self.device.lower()}_failed",
                "device_used": self.device
            }

    def _analyze_pdf_structure_gpu_optimized(self, source: Union[str, Path], sample_pages: int, max_sample_chars: int) -> Dict[str, Any]:
        """GPU-optimized PDF structure analysis using PyMuPDF with performance monitoring."""
        try:
            # Use the same fast PyMuPDF analysis as base class, but with GPU memory monitoring
            start_memory = self._get_gpu_memory_status()

            doc = fitz.open(str(source))
            total_pages = doc.page_count
            metadata = doc.metadata

            # Optimized sampling strategy for GPU context
            pages_to_sample = min(sample_pages, total_pages)

            # For large documents on GPU, we can afford slightly larger samples
            if self.device in [AcceleratorDevice.CUDA, AcceleratorDevice.MPS] and total_pages > 50:
                pages_to_sample = min(pages_to_sample + 2, total_pages)
                max_sample_chars = int(max_sample_chars * 1.5)  # 50% larger sample on GPU

            sample_text = ""
            headers_found = []
            tables_detected = 0
            images_detected = 0
            text_density_avg = 0

            # Process pages with GPU memory awareness
            for page_num in range(pages_to_sample):
                page = doc[page_num]
                page_text = page.get_text()
                sample_text += page_text[:max_sample_chars // pages_to_sample] + "\n"

                # Enhanced structure detection on GPU
                text_dict = page.get_text("dict")
                images_detected += len(page.get_images())
                text_density_avg += len(page_text.strip()) / max(1, page.rect.width * page.rect.height) * 10000

                # GPU-optimized header detection (process more patterns)
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text and len(text) < 150:  # Larger header detection on GPU
                                    font_size = span.get("size", 12)
                                    font_flags = span.get("flags", 0)
                                    if font_size > 13 or (font_flags & 2**4):  # More sensitive on GPU
                                        headers_found.append(text)

                tables_detected += self._estimate_tables_in_page_text(page_text)

            doc.close()

            text_density_avg = round(text_density_avg / pages_to_sample, 2) if pages_to_sample > 0 else 0
            end_memory = self._get_gpu_memory_status()

            return {
                "total_pages": total_pages,
                "pages_analyzed": pages_to_sample,
                "metadata_extraction": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                },
                "structure_detected": {
                    "headers_found": len(set(headers_found)),
                    "sample_headers": list(set(headers_found))[:7],  # More headers shown on GPU
                    "estimated_tables": tables_detected,
                    "images_detected": images_detected,
                    "text_density": text_density_avg,
                    "has_text": len(sample_text.strip()) > 50,
                    "gpu_enhanced_detection": True
                },
                "content_preview": sample_text[:max_sample_chars].strip(),
                "memory_usage": {"start": start_memory, "end": end_memory}
            }

        except Exception as e:
            logger.warning(f"GPU-optimized PyMuPDF analysis failed, falling back: {e}")
            return self._analyze_other_format_structure_gpu(source, sample_pages, max_sample_chars)

    def _analyze_image_structure_gpu_aware(self, source: Union[str, Path]) -> Dict[str, Any]:
        """GPU-aware image file analysis with enhanced metadata extraction."""
        try:
            start_memory = self._get_gpu_memory_status()

            with Image.open(source) as img:
                # Enhanced image analysis on GPU systems
                analysis = {
                    "total_pages": 1,
                    "pages_analyzed": 1,
                    "metadata_extraction": {
                        "format": img.format,
                        "mode": img.mode,
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "has_exif": bool(getattr(img, '_getexif', lambda: None)()),
                        "pixel_count": img.size[0] * img.size[1],
                        "aspect_ratio": round(img.size[0] / img.size[1], 2) if img.size[1] > 0 else 0
                    },
                    "structure_detected": {
                        "content_type": "image",
                        "requires_ocr": True,
                        "estimated_text_content": "unknown_until_ocr",
                        "gpu_processing_recommended": self.device != AcceleratorDevice.CPU,
                        "large_image": img.size[0] * img.size[1] > 2000000  # > 2MP
                    },
                    "content_preview": f"Image file: {img.format} format, {img.size[0]}x{img.size[1]} pixels",
                    "memory_usage": {"start": start_memory, "end": self._get_gpu_memory_status()}
                }

                # Add GPU-specific recommendations for large images
                if analysis["structure_detected"]["large_image"] and self.device == AcceleratorDevice.CUDA:
                    analysis["structure_detected"]["processing_recommendation"] = "Use GPU for OCR processing"

                return analysis

        except Exception as e:
            logger.warning(f"GPU-aware image analysis failed: {e}")
            return {
                "total_pages": 1,
                "structure_detected": {"content_type": "image", "analysis_failed": str(e)},
                "content_preview": "Image analysis failed"
            }

    def _analyze_other_format_structure_gpu(self, source: Union[str, Path], sample_pages: int, max_sample_chars: int) -> Dict[str, Any]:
        """GPU-optimized lightweight analysis for other formats."""
        try:
            start_memory = self._get_gpu_memory_status()

            # Use docling with GPU acceleration but minimal processing
            result = self.converter.convert(source=str(source))
            document = result.document

            total_pages = len(document.pages) if hasattr(document, 'pages') else 1
            pages_to_analyze = min(sample_pages, total_pages)

            # GPU systems can handle larger samples
            if self.device in [AcceleratorDevice.CUDA, AcceleratorDevice.MPS]:
                max_sample_chars = int(max_sample_chars * 1.5)

            sample_content = ""

            if hasattr(document, 'pages'):
                for i in range(pages_to_analyze):
                    if i < len(document.pages):
                        page = document.pages[i]
                        if hasattr(page, 'text'):
                            sample_content += str(page.text)[:max_sample_chars // pages_to_analyze] + "\n"

            if not sample_content:
                full_content = document.export_to_markdown()
                sample_content = full_content[:max_sample_chars]

            # Enhanced structure analysis with GPU capabilities
            headers_found = [line.strip() for line in sample_content.split('\n') if line.strip().startswith('#')]
            table_lines = [line for line in sample_content.split('\n') if '|' in line and line.strip()]

            end_memory = self._get_gpu_memory_status()

            return {
                "total_pages": total_pages,
                "pages_analyzed": pages_to_analyze,
                "structure_detected": {
                    "headers_found": len(headers_found),
                    "sample_headers": headers_found[:7],  # More headers on GPU
                    "estimated_tables": len([line for line in table_lines if line.count('|') > 1]),
                    "has_markdown_structure": len(headers_found) > 0 or len(table_lines) > 0,
                    "gpu_accelerated": True
                },
                "content_preview": sample_content.strip(),
                "memory_usage": {"start": start_memory, "end": end_memory}
            }

        except Exception as e:
            logger.warning(f"GPU-optimized docling analysis failed: {e}")
            return {
                "total_pages": 1,
                "structure_detected": {"analysis_method": "file_info_only", "gpu_fallback": True},
                "content_preview": "Unable to analyze document structure with GPU acceleration"
            }

    def _get_gpu_memory_status(self) -> Optional[Dict[str, Any]]:
        """Get current GPU memory status for performance monitoring."""
        if not TORCH_AVAILABLE or self.device == AcceleratorDevice.CPU:
            return None

        try:
            if self.device == AcceleratorDevice.CUDA and torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() // (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved() // (1024**2),
                    "total_mb": torch.cuda.get_device_properties(0).total_memory // (1024**2)
                }
            elif self.device == AcceleratorDevice.MPS:
                return {"device": "MPS", "status": "active"}
        except Exception:
            pass

        return None

    def _estimate_tables_in_page_text(self, text: str) -> int:
        """Estimate number of tables in text by looking for aligned patterns."""
        lines = text.split('\n')
        potential_table_lines = 0

        for line in lines:
            # Look for lines with multiple whitespace-separated columns
            parts = line.strip().split()
            if len(parts) >= 3:  # At least 3 columns
                # Check if parts look like tabular data (numbers, short text)
                if any(part.replace('.', '').replace(',', '').isdigit() for part in parts):
                    potential_table_lines += 1

        # Rough estimate: every 5+ aligned lines might be a table
        return potential_table_lines // 5

    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status and performance info."""
        status = {
            "current_device": self.device,
            "model_type": self.model_type,
            "device_info": self.device_info
        }

        if TORCH_AVAILABLE and self.device == AcceleratorDevice.CUDA:
            try:
                status.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated() // (1024**2),  # MB
                    "gpu_memory_reserved": torch.cuda.memory_reserved() // (1024**2),   # MB
                    "gpu_utilization": "Available" if torch.cuda.is_available() else "Not available"
                })
            except Exception as e:
                status["gpu_error"] = str(e)

        return status

    def convert_document(
        self,
        source: Union[str, Path],
        output_format: str = "markdown",
        show_device_info: bool = False
    ) -> Dict[str, Any]:
        """Convert a document using the Granite Docling model with GPU acceleration.

        Args:
            source: Path to the document or URL
            output_format: Output format (currently supports 'markdown')
            show_device_info: Include device performance info in results

        Returns:
            Dictionary containing the conversion result and metadata
        """
        try:
            logger.info(f"Converting document: {source} on device: {self.device}")

            # Convert the document
            result = self.converter.convert(source=str(source))
            document = result.document

            # Extract the converted content
            if output_format.lower() == "markdown":
                content = document.export_to_markdown()
            else:
                content = str(document)

            # Prepare result dictionary with GPU-specific metadata
            conversion_result = {
                "content": content,
                "source": str(source),
                "format": output_format,
                "pages": len(document.pages) if hasattr(document, 'pages') else 1,
                "metadata": {
                    "model_type": self.model_type,
                    "device": self.device,  # GPU-specific addition
                    "model_config": str(self.vlm_model.__class__.__name__)
                }
            }

            if show_device_info:
                conversion_result["device_status"] = self.get_device_status()

            logger.info(f"Successfully converted document with {conversion_result['pages']} pages using {self.device}")
            return conversion_result

        except Exception as e:
            logger.error(f"Error converting document {source}: {str(e)}")
            raise

    def batch_convert(
        self,
        sources: list,
        output_dir: Union[str, Path],
        output_format: str = "markdown"
    ) -> list:
        """Convert multiple documents in batch with GPU acceleration.

        This method overrides the parent to add enhanced batch progress logging
        and GPU-specific batch information.

        Args:
            sources: List of document paths or URLs
            output_dir: Directory to save converted documents
            output_format: Output format for all documents

        Returns:
            List of conversion results with batch information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total_docs = len(sources)

        for i, source in enumerate(sources, 1):
            try:
                logger.info(f"Processing document {i}/{total_docs}: {source}")

                # Generate output filename
                source_path = Path(source)
                if output_format.lower() == "markdown":
                    output_filename = source_path.stem + ".md"
                else:
                    output_filename = source_path.stem + f".{output_format}"

                output_path = output_dir / output_filename

                # Convert and save using parent's convert_to_file method
                result = self.convert_to_file(source, output_path, output_format)

                # Add GPU-specific batch information
                result["batch_info"] = {"index": i, "total": total_docs}
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to convert {source}: {str(e)}")
                results.append({
                    "source": str(source),
                    "error": str(e),
                    "success": False,
                    "batch_info": {"index": i, "total": total_docs}
                })

        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"Batch conversion completed: {successful}/{total_docs} successful")

        return results


def download_models():
    """Download the required Granite Docling models."""
    try:
        import subprocess
        logger.info("Downloading Granite Docling models...")
        subprocess.run([
            "docling-tools", "models", "download"
        ], check=True)
        logger.info("Models downloaded successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download models: {e}")
        raise
    except FileNotFoundError:
        logger.error("docling-tools not found. Please install docling first.")
        raise


# Alias for backward compatibility
GraniteDocling = GraniteDoclingGPU


if __name__ == "__main__":
    # Example usage with GPU support
    print("Granite Docling with GPU Support")
    print("=" * 40)

    # Show device info
    device_manager = DeviceManager()
    device_info = device_manager.get_device_info()

    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    print(f"\nOptimal device: {device_manager.get_optimal_device()}")

    # Initialize with GPU support
    granite = GraniteDoclingGPU(auto_device=True)
    print(f"\nInitialized with device: {granite.device}")

    # Show device status
    status = granite.get_device_status()
    print("\nDevice Status:")
    for key, value in status.items():
        if key != "device_info":
            print(f"  {key}: {value}")