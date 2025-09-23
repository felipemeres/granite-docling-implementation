# Granite Docling 258M - Interactive Web Demo

This directory contains interactive web interfaces for demonstrating the Granite Docling 258M model capabilities.

## 🚀 Quick Start

### Option 1: Windows (Double-click)
```bash
# Just double-click the batch file:
run_demo.bat
```

### Option 2: Command Line
```bash
# Check setup first
python check_setup.py

# Run simple demo
python simple_demo.py

# Or run full-featured demo
python gradio_interface.py
```

### Option 3: Cross-platform
```bash
# Linux/Mac
chmod +x run_demo.sh
./run_demo.sh

# Windows
run_demo.bat
```

## 📁 Demo Files

| File | Description |
|------|-------------|
| `simple_demo.py` | Basic interface for testing setup |
| `gradio_interface.py` | Full-featured demo with all processing modes |
| `check_setup.py` | Verify installation and setup |
| `run_demo.bat` | Windows launcher script |
| `run_demo.sh` | Linux/Mac launcher script |

## 🎯 Features Demonstrated

### Processing Modes
1. **Full Markdown Conversion** - Complete document → Markdown
2. **Document Analysis** - Structure analysis and metadata
3. **Table Extraction** - Focus on extracting table data
4. **Quick Preview** - Fast preview of document content

### Model Backends
- **Transformers** (Default) - Works on all systems
- **MLX** - Optimized for Apple Silicon Macs

### File Support
- 📄 PDF documents
- 📝 Word documents (.docx, .doc)
- 🖼️ Images (.png, .jpg, .jpeg)

## 🛠️ Setup Requirements

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
```bash
docling-tools models download
```

### 3. Verify Setup
```bash
python check_setup.py
```

## 🌐 Interface Overview

### Simple Demo (`simple_demo.py`)
- **URL**: http://127.0.0.1:7860
- **Purpose**: Basic setup testing
- **Features**: File upload, model selection, status check

### Full Demo (`gradio_interface.py`)
- **URL**: http://127.0.0.1:7861 (or as configured)
- **Purpose**: Complete feature demonstration
- **Features**:
  - Multiple processing modes
  - Real-time processing
  - Metadata display
  - Error handling
  - Results in multiple formats

## 📊 Usage Examples

### Basic Document Processing
1. Upload a PDF or image file
2. Select "Full Markdown Conversion"
3. Choose "transformers" backend
4. Click "Process Document"
5. View results in the output tabs

### Table Extraction
1. Upload a document with tables
2. Select "Table Extraction" mode
3. Process and view extracted table data

### Document Analysis
1. Upload any document
2. Select "Document Analysis" mode
3. View structure analysis and metadata

## 🔧 Troubleshooting

### Common Issues

**"Docling not available"**
```bash
pip install -r requirements.txt
```

**"Models not found"**
```bash
docling-tools models download
```

**"Permission denied" (Linux/Mac)**
```bash
chmod +x run_demo.sh
```

**Port already in use**
- Default ports: 7860 (simple), 7861 (full)
- Change port in the script if needed

### Performance Tips
- Use GPU if available for faster processing
- Start with small documents for testing
- MLX backend is faster on Apple Silicon
- Allow time for model initialization on first run

## 🚀 Deployment Options

### Local Development
```bash
python simple_demo.py
# Access at http://127.0.0.1:7860
```

### Network Access
```python
# Modify in the script:
demo.launch(
    server_name="0.0.0.0",  # Allow network access
    server_port=7860,
    share=False
)
# Access at http://YOUR_IP:7860
```

### Public Sharing (Temporary)
```python
# Modify in the script:
demo.launch(share=True)  # Creates public Gradio link
```

## 📝 Customization

### Adding New Processing Modes
Edit `gradio_interface.py` and add to the `process_document` method:

```python
elif processing_mode == "Your Custom Mode":
    # Your custom processing logic
    result = granite.convert_document(temp_file)
    # Custom post-processing
    markdown_output = custom_process(result["content"])
```

### Changing Interface Layout
Modify the `create_interface` method in `GraniteDoclingInterface` class.

### Custom Styling
Update the CSS in the `create_interface` method.

## 🎉 Success Indicators

✅ **Setup Working**: `check_setup.py` shows all OK
✅ **Demo Running**: Browser opens to interface
✅ **Processing Working**: Documents convert to Markdown
✅ **Multiple Modes**: All processing modes function

## 📞 Support

If you encounter issues:
1. Run `python check_setup.py` first
2. Check the console output for error messages
3. Verify all dependencies are installed
4. Ensure models are downloaded

The demo interfaces provide a comprehensive way to explore and demonstrate the capabilities of the Granite Docling 258M model!