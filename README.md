# Granite Docling 258M Implementation

This repository implements the IBM Granite Docling 258M model for advanced document processing and conversion.

## Overview

The IBM Granite Docling 258M is a Vision-Language Model (VLM) designed for document understanding and conversion tasks. It can process various document formats including PDF, DOCX, and images, converting them to structured formats like Markdown while maintaining semantic understanding.

## Features

- Document processing with vision-language understanding
- Support for multiple input formats (PDF, DOCX, images)
- Markdown conversion with semantic preservation
- Picture description and content extraction
- Table structure recognition
- OCR capabilities

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd granite-docling-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Granite Docling models:
```bash
docling-tools models download-hf-repo ibm-granite/granite-docling-258M
```

## Usage

See `examples/` directory for usage examples.

## Model Information

- **Model**: IBM Granite Docling 258M
- **Type**: Vision-Language Model
- **Primary Use**: Document processing and understanding
- **HuggingFace**: https://huggingface.co/ibm-granite/granite-docling-258M

## License

This project follows the same licensing terms as the original IBM Granite Docling model.