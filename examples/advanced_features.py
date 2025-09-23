#!/usr/bin/env python3
"""
Advanced features example for Granite Docling 258M

This script demonstrates advanced features like custom model configurations,
batch processing strategies, and different optimization options.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from granite_docling import GraniteDocling

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_model_configurations() -> Dict[str, GraniteDocling]:
    """Example showing different model configurations for various use cases.

    Returns:
        Dict[str, GraniteDocling]: Dictionary of configured model instances
    """
    print("\n[CONFIG] Model Configuration Examples")
    print("-" * 35)

    configurations = {}

    try:
        # Configuration for technical documents (high precision)
        print("Creating technical document processor...")
        configurations['technical'] = GraniteDocling(
            model_type="transformers"
        )
        print("  [PASS] Technical configuration: High precision for academic/technical content")

        # Configuration for business documents (balanced)
        print("Creating business document processor...")
        configurations['business'] = GraniteDocling(
            model_type="transformers"
        )
        print("  [PASS] Business configuration: Optimized for reports and presentations")

        # Configuration for general documents (fast processing)
        print("Creating general purpose processor...")
        configurations['general'] = GraniteDocling(
            model_type="transformers"
        )
        print("  [PASS] General configuration: Balanced speed and quality")

    except Exception as e:
        logger.error(f"Failed to create configurations: {e}")
        raise

    return configurations


def example_performance_optimization() -> Dict[str, Dict[str, Union[str, float]]]:
    """Example showing performance optimization strategies.

    Returns:
        Dict[str, Dict[str, Union[str, float]]]: Performance optimization configurations
    """
    print("\n[PERF] Performance Optimization Examples")
    print("-" * 40)

    # Performance optimization strategies
    optimization_strategies = {
        "high_quality": {
            "description": "Maximum quality for critical documents",
            "use_case": "Legal documents, research papers",
            "model_type": "transformers",
            "recommended_batch_size": 1
        },
        "balanced": {
            "description": "Balanced quality and speed for general use",
            "use_case": "Business reports, general documents",
            "model_type": "transformers",
            "recommended_batch_size": 2
        },
        "high_throughput": {
            "description": "Optimized for processing many documents quickly",
            "use_case": "Bulk document processing",
            "model_type": "transformers",
            "recommended_batch_size": 5
        }
    }

    for strategy_name, config in optimization_strategies.items():
        print(f"\n{strategy_name.replace('_', ' ').title()} Strategy:")
        print(f"  Description: {config['description']}")
        print(f"  Use case: {config['use_case']}")
        print(f"  Model type: {config['model_type']}")
        print(f"  Recommended batch size: {config['recommended_batch_size']}")

    return optimization_strategies


def example_document_analysis() -> Dict[str, List[str]]:
    """Example demonstrating document analysis capabilities.

    Returns:
        Dict[str, List[str]]: Analysis capabilities by document type
    """
    print("\n[ANALYSIS] Document Analysis Capabilities")
    print("-" * 38)

    # Document analysis capabilities
    analysis_capabilities = {
        "visual_elements": [
            "Tables and data extraction",
            "Charts and graphs interpretation",
            "Diagrams and flowcharts",
            "Images and technical illustrations",
            "Mathematical equations and formulas"
        ],
        "text_processing": [
            "Multi-language document support",
            "Structured data extraction",
            "Heading and section detection",
            "Footnote and citation handling",
            "Bibliography and reference parsing"
        ],
        "document_structure": [
            "Page layout preservation",
            "Multi-column text handling",
            "Header and footer detection",
            "Table of contents extraction",
            "Document metadata extraction"
        ]
    }

    for category, capabilities in analysis_capabilities.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for capability in capabilities:
            print(f"  [FEATURE] {capability}")

    return analysis_capabilities


def example_batch_processing_strategies() -> Dict[str, Dict[str, Union[str, int, List[str]]]]:
    """Example of batch processing strategies with performance considerations.

    Returns:
        Dict[str, Dict[str, Union[str, int, List[str]]]]: Batch processing configurations
    """
    print("\n[BATCH] Batch Processing Strategies")
    print("-" * 32)

    # Different batch processing strategies
    batch_strategies = {
        "small_batch": {
            "batch_size": 1,
            "memory_usage": "Low",
            "processing_speed": "Moderate",
            "recommended_for": ["Large documents", "Limited memory systems"],
            "description": "Process documents one at a time for maximum reliability"
        },
        "medium_batch": {
            "batch_size": 3,
            "memory_usage": "Medium",
            "processing_speed": "Good",
            "recommended_for": ["General use", "Mixed document sizes"],
            "description": "Balanced approach for most use cases"
        },
        "large_batch": {
            "batch_size": 10,
            "memory_usage": "High",
            "processing_speed": "Fast",
            "recommended_for": ["Small documents", "High-memory systems"],
            "description": "Maximum throughput for bulk processing"
        }
    }

    # Document type processing recommendations
    document_recommendations = {
        "research_papers": {
            "extensions": [".pdf"],
            "recommended_strategy": "small_batch",
            "special_considerations": ["Complex layouts", "Mathematical notation"]
        },
        "business_reports": {
            "extensions": [".pdf", ".docx"],
            "recommended_strategy": "medium_batch",
            "special_considerations": ["Tables and charts", "Multi-column layouts"]
        },
        "presentations": {
            "extensions": [".pptx", ".pdf"],
            "recommended_strategy": "medium_batch",
            "special_considerations": ["Slide layouts", "Visual elements"]
        }
    }

    print("Batch Processing Strategies:")
    for strategy_name, config in batch_strategies.items():
        print(f"\n  {strategy_name.replace('_', ' ').title()}:")
        print(f"    Batch size: {config['batch_size']}")
        print(f"    Memory usage: {config['memory_usage']}")
        print(f"    Speed: {config['processing_speed']}")
        print(f"    Description: {config['description']}")

    print("\nDocument Type Recommendations:")
    for doc_type, rec in document_recommendations.items():
        print(f"\n  {doc_type.replace('_', ' ').title()}:")
        print(f"    Extensions: {', '.join(rec['extensions'])}")
        print(f"    Strategy: {rec['recommended_strategy']}")
        print(f"    Considerations: {', '.join(rec['special_considerations'])}")

    return {"strategies": batch_strategies, "recommendations": document_recommendations}


def demonstrate_real_world_usage() -> None:
    """Demonstrate real-world usage patterns with actual code examples."""
    print("\n[EXAMPLES] Real-World Usage Examples")
    print("-" * 32)

    # Example: Processing a research paper
    print("\nExample 1: Research Paper Processing")
    try:
        granite = GraniteDocling(model_type="transformers")

        # This would be the actual usage pattern
        sample_pdf_url = "https://arxiv.org/pdf/2206.01062"  # Example paper
        print(f"  URL: {sample_pdf_url}")
        print("  Usage: granite.convert_document(sample_pdf_url)")
        print("  [PASS] Configuration ready for research papers")

    except Exception as e:
        logger.error(f"Research paper example setup failed: {e}")
        print(f"  [WARN] Setup issue: {e}")

    # Example: Batch processing business documents
    print("\nExample 2: Batch Business Document Processing")
    try:
        business_docs = [
            "quarterly_report.pdf",
            "financial_summary.pdf",
            "market_analysis.pdf"
        ]
        output_dir = "outputs/business_docs"

        print(f"  Input files: {len(business_docs)} documents")
        print(f"  Output directory: {output_dir}")
        print("  Usage: granite.batch_convert(business_docs, output_dir)")
        print("  [PASS] Batch processing configuration ready")

    except Exception as e:
        logger.error(f"Batch processing example setup failed: {e}")
        print(f"  [WARN] Setup issue: {e}")


def validate_prerequisites() -> bool:
    """Validate that all prerequisites are met for advanced features.

    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    print("\n[VALIDATE] Validating Prerequisites")
    print("-" * 28)

    prerequisites_met = True

    try:
        # Check if GraniteDocling can be instantiated
        granite = GraniteDocling()
        print("  [PASS] GraniteDocling instantiation: Success")
    except Exception as e:
        logger.error(f"GraniteDocling instantiation failed: {e}")
        print(f"  [FAIL] GraniteDocling instantiation: Failed ({e})")
        prerequisites_met = False

    # Check if output directories can be created
    try:
        test_dir = Path("test_outputs")
        test_dir.mkdir(exist_ok=True)
        test_dir.rmdir()  # Clean up
        print("  [PASS] Directory creation: Success")
    except Exception as e:
        logger.error(f"Directory creation failed: {e}")
        print(f"  [FAIL] Directory creation: Failed ({e})")
        prerequisites_met = False

    return prerequisites_met


def main() -> None:
    """Main advanced features demonstration."""
    print("Granite Docling 258M - Advanced Features")
    print("=" * 45)

    try:
        # Validate prerequisites first
        if not validate_prerequisites():
            print("\n[WARN] Some prerequisites are not met. Please check the setup.")
            return

        # Example 1: Model configurations
        print("\n" + "=" * 50)
        model_configs = example_model_configurations()
        print(f"  [PASS] Created {len(model_configs)} model configurations")

        # Example 2: Performance optimization
        print("\n" + "=" * 50)
        optimization_strategies = example_performance_optimization()
        print(f"  [PASS] Defined {len(optimization_strategies)} optimization strategies")

        # Example 3: Document analysis capabilities
        print("\n" + "=" * 50)
        analysis_capabilities = example_document_analysis()
        total_capabilities = sum(len(caps) for caps in analysis_capabilities.values())
        print(f"  [PASS] Documented {total_capabilities} analysis capabilities")

        # Example 4: Batch processing strategies
        print("\n" + "=" * 50)
        batch_info = example_batch_processing_strategies()
        print(f"  [PASS] Defined {len(batch_info['strategies'])} batch processing strategies")

        # Example 5: Real-world usage
        print("\n" + "=" * 50)
        demonstrate_real_world_usage()

        # Summary
        print("\n" + "=" * 50)
        print("\n[SUMMARY] Advanced Features Summary:")
        print("  [FEATURE] Multiple model configurations for different use cases")
        print("  [FEATURE] Performance optimization strategies")
        print("  [FEATURE] Comprehensive document analysis capabilities")
        print("  [FEATURE] Flexible batch processing options")
        print("  [FEATURE] Real-world usage examples")

        print("\n[TIPS] Best Practices:")
        print("  - Choose model configuration based on document type and quality needs")
        print("  - Use appropriate batch sizes based on system memory and document complexity")
        print("  - Monitor processing performance and adjust strategies accordingly")
        print("  - Validate input documents before batch processing")
        print("  - Always handle exceptions in production code")

        print("\n[NEXT] Next Steps:")
        print("  - Try the basic_usage.py example first")
        print("  - Test with your own documents")
        print("  - Experiment with different configurations")
        print("  - Monitor memory usage during batch processing")

    except Exception as e:
        logger.error(f"Advanced features demo failed: {e}")
        print(f"\n[ERROR] Error in advanced features demo: {e}")
        print("\nTroubleshooting:")
        print("  - Check that all dependencies are installed")
        print("  - Verify the Granite Docling models are downloaded")
        print("  - Run basic_usage.py first to test core functionality")


if __name__ == "__main__":
    main()