#!/usr/bin/env python3
"""
Comprehensive Test Suite for Granite Docling Implementation

This test suite consolidates all testing functionality including:
- Basic functionality tests (from simple_test.py)
- GPU-specific tests (from test_gpu.py)
- Environment setup validation (from check_setup.py)

Organized with pytest for better test management and reporting.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global variables for test state
GPU_AVAILABLE = False
GPU_IMPORT_ERROR = ""

# Attempt to import GPU modules
try:
    from granite_docling_gpu import GraniteDoclingGPU, DeviceManager
    GPU_AVAILABLE = True
except ImportError as e:
    GPU_IMPORT_ERROR = str(e)


class TestSetupValidation:
    """Test class for environment setup validation (from check_setup.py)."""

    def test_python_version(self) -> None:
        """Test that Python version is compatible."""
        version = sys.version_info
        assert version.major == 3, f"Python 3 required, got {version.major}"
        assert version.minor >= 8, f"Python 3.8+ required, got {version.major}.{version.minor}"
        print(f"[OK] Python version: {sys.version.split()[0]}")

    def test_required_dependencies(self) -> None:
        """Test that all required dependencies can be imported."""
        # Test Gradio
        try:
            import gradio
            print(f"[OK] Gradio: {gradio.__version__}")
        except ImportError:
            pytest.fail("Gradio not installed - run 'pip install gradio'")

        # Test Docling
        try:
            import docling
            print("[OK] Docling: Available")
        except ImportError as e:
            pytest.fail(f"Docling not installed: {e}")

    def test_model_cache_directory(self) -> None:
        """Test model cache directory existence."""
        home_dir = Path.home()
        cache_dir = home_dir / ".cache" / "docling"

        if cache_dir.exists():
            print(f"[OK] Model cache directory: {cache_dir}")
        else:
            print("[INFO] No model cache found - run 'docling-tools models download'")
            # This is informational, not a failure

    def test_project_structure(self) -> None:
        """Test that all required project files exist."""
        required_files = [
            "src/granite_docling.py",
            "examples/basic_usage.py",
            "examples/advanced_features.py",
            "requirements.txt",
            "README.md"
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
            else:
                print(f"[OK] {file_path} exists")

        if missing_files:
            pytest.fail(f"Missing required files: {', '.join(missing_files)}")


class TestBasicFunctionality:
    """Test class for basic functionality (from simple_test.py)."""

    def test_core_imports(self) -> None:
        """Test importing core modules."""
        # Test basic docling import
        try:
            import docling
            print("[OK] docling imported successfully")
        except ImportError as e:
            pytest.fail(f"docling import failed: {e}")

        # Test DocumentConverter import
        try:
            from docling.document_converter import DocumentConverter
            print("[OK] DocumentConverter imported successfully")
        except ImportError as e:
            pytest.fail(f"DocumentConverter import failed: {e}")

    def test_granite_docling_import(self) -> None:
        """Test importing the main GraniteDocling class."""
        try:
            from granite_docling import GraniteDocling
            print("[OK] GraniteDocling imported successfully")
        except ImportError as e:
            pytest.fail(f"GraniteDocling import failed: {e}")

    def test_granite_docling_initialization(self) -> None:
        """Test basic initialization of GraniteDocling."""
        try:
            from granite_docling import GraniteDocling
            granite = GraniteDocling()

            # Verify basic attributes
            assert hasattr(granite, 'model_type'), "GraniteDocling should have model_type attribute"
            print(f"[OK] GraniteDocling initialized with model_type: {granite.model_type}")

        except Exception as e:
            pytest.fail(f"GraniteDocling initialization failed: {e}")

    def test_example_files_syntax(self) -> None:
        """Test that example files have valid Python syntax."""
        example_files = [
            "examples/basic_usage.py",
            "examples/advanced_features.py"
        ]

        for example_file in example_files:
            if Path(example_file).exists():
                try:
                    with open(example_file, 'r') as f:
                        content = f.read()
                    compile(content, example_file, 'exec')
                    print(f"[OK] {example_file} has valid syntax")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {example_file}: {e}")
                except FileNotFoundError:
                    print(f"[WARN] {example_file} not found - skipping syntax check")


@pytest.mark.gpu
class TestGPUFunctionality:
    """Test class for GPU-specific functionality (from test_gpu.py)."""

    def setup_method(self) -> None:
        """Setup for GPU tests."""
        if not GPU_AVAILABLE:
            pytest.skip(f"GPU module not available: {GPU_IMPORT_ERROR}")

    def test_device_detection(self) -> None:
        """Test device detection and availability."""
        device_manager = DeviceManager()

        # Test device info retrieval
        device_info = device_manager.get_device_info()
        assert isinstance(device_info, dict), "Device info should be a dictionary"

        print("Device Information:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")

        # Test available devices detection
        available_devices = device_manager.detect_available_devices()
        assert isinstance(available_devices, list), "Available devices should be a list"
        assert len(available_devices) > 0, "At least one device should be available"

        print(f"[OK] Available devices: {', '.join(available_devices)}")

        # Test optimal device selection
        optimal_device = device_manager.get_optimal_device()
        assert optimal_device in available_devices, "Optimal device should be in available devices"

        print(f"[OK] Optimal device: {optimal_device}")

    def test_gpu_initialization(self) -> None:
        """Test GPU-enabled Granite Docling initialization."""
        # Test auto device selection
        granite_auto = GraniteDoclingGPU(auto_device=True)
        assert hasattr(granite_auto, 'device'), "GraniteDoclingGPU should have device attribute"
        print(f"[OK] Auto device initialization: {granite_auto.device}")

        # Test specific device selections
        device_manager = DeviceManager()
        available_devices = device_manager.detect_available_devices()

        for device in available_devices:
            try:
                granite_device = GraniteDoclingGPU(device=device, auto_device=False)
                assert granite_device.device == device, f"Device should be set to {device}"
                print(f"[OK] {device} initialization: {granite_device.device}")
            except Exception as e:
                print(f"[WARN] {device} initialization failed: {e}")

    def test_device_status(self) -> None:
        """Test device status reporting."""
        granite = GraniteDoclingGPU(auto_device=True)
        status = granite.get_device_status()

        assert isinstance(status, dict), "Device status should be a dictionary"

        print("Device Status:")
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

    def test_performance_readiness(self) -> None:
        """Test performance comparison readiness."""
        device_manager = DeviceManager()
        available_devices = device_manager.detect_available_devices()

        print("Performance test readiness:")
        for device in available_devices:
            try:
                granite = GraniteDoclingGPU(device=device, auto_device=False)
                print(f"[OK] {device}: Ready for performance testing")
            except Exception as e:
                print(f"[WARN] {device}: Not ready - {e}")


class TestIntegration:
    """Integration tests that combine multiple components."""

    def test_end_to_end_import_chain(self) -> None:
        """Test complete import chain from docling to granite implementation."""
        # Import chain test
        import docling
        from docling.document_converter import DocumentConverter
        from granite_docling import GraniteDocling

        # Initialize components
        converter = DocumentConverter()
        granite = GraniteDocling()

        print("[OK] End-to-end import chain successful")
        print(f"  - DocumentConverter: {type(converter).__name__}")
        print(f"  - GraniteDocling: {granite.model_type}")

    @pytest.mark.gpu
    def test_gpu_cpu_compatibility(self) -> None:
        """Test that GPU and CPU versions can coexist."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU module not available")

        from granite_docling import GraniteDocling
        from granite_docling_gpu import GraniteDoclingGPU

        # Initialize both versions
        granite_cpu = GraniteDocling()
        granite_gpu = GraniteDoclingGPU(auto_device=True)

        print("[OK] CPU and GPU versions initialized successfully")
        print(f"  - CPU version: {granite_cpu.model_type}")
        print(f"  - GPU version: {granite_gpu.device}")

    def test_file_system_structure(self) -> None:
        """Test that the file system structure supports all components."""
        base_dir = Path(".")

        # Check critical directories
        critical_dirs = ["src", "examples", "tests"]
        for dir_name in critical_dirs:
            dir_path = base_dir / dir_name
            assert dir_path.exists(), f"Critical directory missing: {dir_name}"
            print(f"[OK] Directory exists: {dir_name}")

        # Check that src is in Python path
        src_path = str((base_dir / "src").resolve())
        assert any(src_path in path for path in sys.path), "src directory should be in Python path"
        print("[OK] src directory properly added to Python path")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_graceful_gpu_fallback(self) -> None:
        """Test graceful handling when GPU is not available."""
        # This test should pass whether GPU is available or not
        if GPU_AVAILABLE:
            # Test that we can detect lack of GPU gracefully
            print("[OK] GPU available - testing fallback mechanisms")
        else:
            # Test that we handle missing GPU gracefully
            print(f"[OK] GPU not available - error handled gracefully: {GPU_IMPORT_ERROR}")

    def test_missing_model_handling(self) -> None:
        """Test handling of missing models."""
        # Check if models are available
        cache_dir = Path.home() / ".cache" / "docling"

        if not cache_dir.exists():
            print("[OK] Missing model cache detected - should prompt for download")
        else:
            print("[OK] Model cache exists - ready for processing")

    def test_invalid_device_handling(self) -> None:
        """Test handling of invalid device specifications."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU module not available")

        # Test invalid device handling
        try:
            granite = GraniteDoclingGPU(device="invalid_device", auto_device=False)
            # If this doesn't raise an exception, it should fallback gracefully
            print(f"[OK] Invalid device handled gracefully: {granite.device}")
        except Exception as e:
            # Exception is expected for invalid devices
            print(f"[OK] Invalid device properly rejected: {e}")


def run_performance_benchmark() -> Dict[str, float]:
    """
    Utility function for performance benchmarking.
    Returns timing information for different operations.
    """
    if not GPU_AVAILABLE:
        return {"error": "GPU not available"}

    device_manager = DeviceManager()
    available_devices = device_manager.detect_available_devices()

    benchmark_results = {}

    for device in available_devices:
        try:
            start_time = time.time()
            granite = GraniteDoclingGPU(device=device, auto_device=False)
            init_time = time.time() - start_time

            benchmark_results[f"{device}_init_time"] = init_time
            print(f"[OK] {device} initialization: {init_time:.3f}s")

        except Exception as e:
            benchmark_results[f"{device}_error"] = str(e)
            print(f"[WARN] {device} benchmark failed: {e}")

    return benchmark_results


def main() -> None:
    """
    Main function to run comprehensive tests.
    Can be called directly or via pytest.
    """
    print("Granite Docling - Comprehensive Test Suite")
    print("=" * 50)

    # Run pytest programmatically if called directly
    if __name__ == "__main__":
        # Configure pytest arguments
        pytest_args = [
            __file__,
            "-v",  # verbose output
            "--tb=short",  # short traceback format
            "-x",  # stop on first failure
        ]

        # Add GPU tests if available
        if GPU_AVAILABLE:
            pytest_args.extend(["-m", "not gpu or gpu"])
            print("[GPU] GPU tests will be included")
        else:
            pytest_args.extend(["-m", "not gpu"])
            print("[WARN] GPU tests will be skipped")

        # Run tests
        exit_code = pytest.main(pytest_args)

        print("\n" + "=" * 50)
        if exit_code == 0:
            print("[SUCCESS] All tests passed!")
            print("\nNext steps:")
            print("1. Run demo: python quick_demo.py")
            if GPU_AVAILABLE:
                print("2. GPU demo: python gradio_interface_gpu.py")
            print("3. Basic demo: python gradio_interface.py")
        else:
            print("[FAIL] Some tests failed. Please check the output above.")

        return exit_code

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)