"""
Granite Docling 258M Implementation

This module provides an interface to the IBM Granite Docling 258M model
for document processing and conversion tasks.
"""

import os
import logging
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