"""
Granite Docling 258M Implementation

This module provides an interface to the IBM Granite Docling 258M model
for document processing and conversion tasks.
"""

import os
import logging
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    VlmPipelineOptions,
    ResponseFormat,
    AcceleratorDevice,
    vlm_model_specs
)
from docling.pipeline.vlm_pipeline import VlmPipeline

# Additional imports for fast document analysis
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraniteDocling:
    """
    A wrapper class for the IBM Granite Docling 258M model.

    This class provides an easy-to-use interface for document processing
    using the Granite Docling model through the Docling framework.
    """

    def __init__(
        self,
        model_type: str = "transformers",
        artifacts_path: Optional[str] = None
    ):
        """
        Initialize the Granite Docling processor.

        Args:
            model_type: Model type - "transformers" or "mlx"
            artifacts_path: Path to cached model artifacts
        """
        self.model_type = model_type.lower()
        self.artifacts_path = artifacts_path

        # Choose the appropriate model configuration
        if self.model_type == "mlx":
            self.vlm_model = vlm_model_specs.GRANITEDOCLING_MLX
        else:
            self.vlm_model = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS

        # Initialize the document converter
        self._setup_converter()

    def _setup_converter(self):
        """Set up the document converter with Granite Docling configuration."""

        # Set up VLM pipeline options using the pre-configured Granite Docling model
        pipeline_options = VlmPipelineOptions(vlm_options=self.vlm_model)

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

        logger.info(f"Initialized Granite Docling with model type: {self.model_type}")

    def analyze_document_structure(
        self,
        source: Union[str, Path],
        sample_pages: int = 3,
        max_sample_chars: int = 2000
    ) -> Dict[str, Any]:
        """
        Fast document structure analysis without full conversion.

        This method provides lightweight document insights including:
        - Basic metadata (pages, size, type)
        - Structure detection (headers, tables, images)
        - Content sampling from first few pages
        - Performance optimized for large documents

        Args:
            source: Path to the document
            sample_pages: Number of pages to sample for content analysis
            max_sample_chars: Maximum characters to extract for preview

        Returns:
            Dictionary containing document analysis and structure information
        """
        start_time = time.time()

        try:
            source_path = Path(source)
            logger.info(f"Analyzing document structure: {source}")

            # Initialize analysis result
            analysis_result = {
                "source": str(source),
                "file_name": source_path.name,
                "file_size_mb": round(source_path.stat().st_size / (1024 * 1024), 2),
                "analysis_time_seconds": 0,
                "document_type": source_path.suffix.lower(),
                "structure_detected": {},
                "content_preview": "",
                "metadata_extraction": {},
                "processing_approach": "fast_analysis"
            }

            # PDF-specific fast analysis
            if source_path.suffix.lower() == '.pdf' and PYMUPDF_AVAILABLE:
                analysis_result.update(self._analyze_pdf_structure(source, sample_pages, max_sample_chars))

            # Image file analysis
            elif source_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'] and PIL_AVAILABLE:
                analysis_result.update(self._analyze_image_structure(source))

            # For other formats, use docling but with limited sampling
            else:
                analysis_result.update(self._analyze_other_format_structure(source, sample_pages, max_sample_chars))

            analysis_result["analysis_time_seconds"] = round(time.time() - start_time, 2)

            logger.info(f"Document analysis completed in {analysis_result['analysis_time_seconds']} seconds")
            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing document structure {source}: {str(e)}")
            return {
                "source": str(source),
                "error": str(e),
                "analysis_time_seconds": round(time.time() - start_time, 2),
                "processing_approach": "fast_analysis_failed"
            }

    def _analyze_pdf_structure(self, source: Union[str, Path], sample_pages: int, max_sample_chars: int) -> Dict[str, Any]:
        """Fast PDF structure analysis using PyMuPDF."""
        try:
            doc = fitz.open(str(source))
            total_pages = doc.page_count

            # Extract metadata
            metadata = doc.metadata

            # Sample pages for structure analysis
            pages_to_sample = min(sample_pages, total_pages)
            sample_text = ""
            headers_found = []
            tables_detected = 0
            images_detected = 0
            text_density_avg = 0

            for page_num in range(pages_to_sample):
                page = doc[page_num]

                # Get text content
                page_text = page.get_text()
                sample_text += page_text[:max_sample_chars // pages_to_sample] + "\n"

                # Detect structure elements
                text_dict = page.get_text("dict")

                # Count images
                images_detected += len(page.get_images())

                # Estimate text density
                text_density_avg += len(page_text.strip()) / max(1, page.rect.width * page.rect.height) * 10000

                # Simple header detection (large/bold text)
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text and len(text) < 100:  # Potential header
                                    font_size = span.get("size", 12)
                                    font_flags = span.get("flags", 0)

                                    # Check if text looks like a header (large font or bold)
                                    if font_size > 14 or (font_flags & 2**4):  # Bold flag
                                        headers_found.append(text)

                # Simple table detection (look for aligned text patterns)
                tables_detected += self._estimate_tables_in_page_text(page_text)

            doc.close()

            text_density_avg = round(text_density_avg / pages_to_sample, 2) if pages_to_sample > 0 else 0

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
                    "sample_headers": list(set(headers_found))[:5],
                    "estimated_tables": tables_detected,
                    "images_detected": images_detected,
                    "text_density": text_density_avg,
                    "has_text": len(sample_text.strip()) > 50
                },
                "content_preview": sample_text[:max_sample_chars].strip()
            }

        except Exception as e:
            logger.warning(f"PyMuPDF analysis failed, falling back: {e}")
            return self._analyze_other_format_structure(source, sample_pages, max_sample_chars)

    def _analyze_image_structure(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Fast image file analysis."""
        try:
            with Image.open(source) as img:
                return {
                    "total_pages": 1,
                    "pages_analyzed": 1,
                    "metadata_extraction": {
                        "format": img.format,
                        "mode": img.mode,
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "has_exif": bool(getattr(img, '_getexif', lambda: None)())
                    },
                    "structure_detected": {
                        "content_type": "image",
                        "requires_ocr": True,
                        "estimated_text_content": "unknown_until_ocr"
                    },
                    "content_preview": f"Image file: {img.format} format, {img.size[0]}x{img.size[1]} pixels"
                }
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return {
                "total_pages": 1,
                "structure_detected": {"content_type": "image", "analysis_failed": str(e)},
                "content_preview": "Image analysis failed"
            }

    def _analyze_other_format_structure(self, source: Union[str, Path], sample_pages: int, max_sample_chars: int) -> Dict[str, Any]:
        """Lightweight analysis for other formats using minimal docling processing."""
        try:
            # Use docling but process minimally - just get basic structure
            result = self.converter.convert(source=str(source))
            document = result.document

            # Get basic info without full markdown conversion
            total_pages = len(document.pages) if hasattr(document, 'pages') else 1

            # Sample first few pages only
            pages_to_analyze = min(sample_pages, total_pages)
            sample_content = ""

            if hasattr(document, 'pages'):
                for i in range(pages_to_analyze):
                    if i < len(document.pages):
                        page = document.pages[i]
                        # Get text content from page without full markdown processing
                        if hasattr(page, 'text'):
                            sample_content += str(page.text)[:max_sample_chars // pages_to_analyze] + "\n"

            # If we still don't have content, do a quick markdown export of first portion
            if not sample_content:
                full_content = document.export_to_markdown()
                sample_content = full_content[:max_sample_chars]

            # Quick structure analysis
            headers_found = [line.strip() for line in sample_content.split('\n') if line.strip().startswith('#')]
            table_lines = [line for line in sample_content.split('\n') if '|' in line and line.strip()]

            return {
                "total_pages": total_pages,
                "pages_analyzed": pages_to_analyze,
                "structure_detected": {
                    "headers_found": len(headers_found),
                    "sample_headers": headers_found[:5],
                    "estimated_tables": len([line for line in table_lines if line.count('|') > 1]),
                    "has_markdown_structure": len(headers_found) > 0 or len(table_lines) > 0
                },
                "content_preview": sample_content.strip()
            }

        except Exception as e:
            logger.warning(f"Docling lightweight analysis failed: {e}")
            return {
                "total_pages": 1,
                "structure_detected": {"analysis_method": "file_info_only"},
                "content_preview": "Unable to analyze document structure"
            }

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

    def convert_document(
        self,
        source: Union[str, Path],
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Convert a document using the Granite Docling model.

        Args:
            source: Path to the document or URL
            output_format: Output format (currently supports 'markdown')

        Returns:
            Dictionary containing the conversion result and metadata
        """
        try:
            logger.info(f"Converting document: {source}")

            # Convert the document
            result = self.converter.convert(source=str(source))
            document = result.document

            # Extract the converted content
            if output_format.lower() == "markdown":
                content = document.export_to_markdown()
            else:
                content = str(document)

            # Prepare result dictionary
            conversion_result = {
                "content": content,
                "source": str(source),
                "format": output_format,
                "pages": len(document.pages) if hasattr(document, 'pages') else 1,
                "metadata": {
                    "model_type": self.model_type,
                    "model_config": str(self.vlm_model.__class__.__name__)
                }
            }

            logger.info(f"Successfully converted document with {conversion_result['pages']} pages")
            return conversion_result

        except Exception as e:
            logger.error(f"Error converting document {source}: {str(e)}")
            raise

    def convert_to_file(
        self,
        source: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Convert a document and save the result to a file.

        Args:
            source: Path to the input document or URL
            output_path: Path where the converted document will be saved
            output_format: Output format (currently supports 'markdown')

        Returns:
            Dictionary containing the conversion result and metadata
        """
        # Convert the document
        result = self.convert_document(source, output_format)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["content"])

        result["output_path"] = str(output_path)
        logger.info(f"Saved converted document to: {output_path}")

        return result

    def batch_convert(
        self,
        sources: list,
        output_dir: Union[str, Path],
        output_format: str = "markdown"
    ) -> list:
        """
        Convert multiple documents in batch.

        Args:
            sources: List of document paths or URLs
            output_dir: Directory to save converted documents
            output_format: Output format for all documents

        Returns:
            List of conversion results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for source in sources:
            try:
                # Generate output filename
                source_path = Path(source)
                if output_format.lower() == "markdown":
                    output_filename = source_path.stem + ".md"
                else:
                    output_filename = source_path.stem + f".{output_format}"

                output_path = output_dir / output_filename

                # Convert and save
                result = self.convert_to_file(source, output_path, output_format)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to convert {source}: {str(e)}")
                results.append({
                    "source": str(source),
                    "error": str(e),
                    "success": False
                })

        return results


def download_models():
    """Download the required Granite Docling models."""
    try:
        import subprocess
        logger.info("Downloading Granite Docling models...")
        subprocess.run([
            "docling-tools", "models", "download-hf-repo",
            "ibm-granite/granite-docling-258M"
        ], check=True)
        logger.info("Models downloaded successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download models: {e}")
        raise
    except FileNotFoundError:
        logger.error("docling-tools not found. Please install docling first.")
        raise


if __name__ == "__main__":
    # Example usage
    granite = GraniteDocling()

    # Example conversion (replace with actual document path)
    # result = granite.convert_document("path/to/document.pdf")
    # print(result["content"])