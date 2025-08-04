#!/usr/bin/env python3
"""
Test script for validating the build process and package structure.
"""

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

def test_build_script():
    """Test the build script functionality"""
    print("Testing build script...")
    
    # Test building a single package
    result = subprocess.run([
        sys.executable, "scripts/build_packages.py", 
        "--provider", "lightspeed-inline-agent",
        "--build-dir", "test_build"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Build script failed: {result.stderr}")
        return False
    
    print("✓ Build script executed successfully")
    return True

def test_package_structure(package_name: str, build_dir: Path):
    """Test the structure of a built package"""
    print(f"Testing package structure for {package_name}...")
    
    package_dir = build_dir / package_name
    
    # Check required files exist
    required_files = [
        "pyproject.toml",
        "README.md",
        "src",
        "config"
    ]
    
    for file_path in required_files:
        if not (package_dir / file_path).exists():
            print(f"✗ Missing required file/directory: {file_path}")
            return False
    
    # Check source structure
    src_dir = package_dir / "src" / package_name.replace("-", "_")
    if not src_dir.exists():
        print(f"✗ Missing source directory: {src_dir}")
        return False
    
    # Check for Python files
    py_files = list(src_dir.rglob("*.py"))
    if not py_files:
        print(f"✗ No Python files found in {src_dir}")
        return False
    
    print(f"✓ Package structure valid for {package_name}")
    return True

def test_pyproject_toml(package_name: str, build_dir: Path):
    """Test the pyproject.toml file"""
    print(f"Testing pyproject.toml for {package_name}...")
    
    pyproject_path = build_dir / package_name / "pyproject.toml"
    
    if not pyproject_path.exists():
        print(f"✗ pyproject.toml not found: {pyproject_path}")
        return False
    
    # Read and validate pyproject.toml
    try:
        import tomllib
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        
        # Check required fields
        project = data.get("project", {})
        required_fields = ["name", "version", "description", "dependencies"]
        
        for field in required_fields:
            if field not in project:
                print(f"✗ Missing required field in pyproject.toml: {field}")
                return False
        
        # Validate package name
        if project["name"] != package_name:
            print(f"✗ Package name mismatch: expected {package_name}, got {project['name']}")
            return False
        
        print(f"✓ pyproject.toml valid for {package_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error parsing pyproject.toml: {e}")
        return False

def test_package_build(package_name: str, build_dir: Path):
    """Test building the package"""
    print(f"Testing package build for {package_name}...")
    
    package_dir = build_dir / package_name
    
    try:
        # Change to package directory
        original_dir = os.getcwd()
        os.chdir(package_dir)
        
        # Build the package
        result = subprocess.run([
            sys.executable, "-m", "build"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Package build failed: {result.stderr}")
            return False
        
        # Check for built artifacts
        dist_dir = package_dir / "dist"
        if not dist_dir.exists():
            print(f"✗ No dist directory created")
            return False
        
        artifacts = list(dist_dir.glob("*"))
        if not artifacts:
            print(f"✗ No build artifacts found")
            return False
        
        print(f"✓ Package build successful for {package_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error building package: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def test_package_install(package_name: str, build_dir: Path):
    """Test installing the package"""
    print(f"Testing package installation for {package_name}...")
    
    package_dir = build_dir / package_name
    dist_dir = package_dir / "dist"
    
    # Find wheel file
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print(f"✗ No wheel file found for {package_name}")
        return False
    
    wheel_file = wheel_files[0]
    
    try:
        # Install the package
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", str(wheel_file), "--force-reinstall"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Package installation failed: {result.stderr}")
            return False
        
        # Test import
        module_name = package_name.replace("-", "_")
        try:
            __import__(module_name)
            print(f"✓ Package installation and import successful for {package_name}")
            return True
        except ImportError as e:
            print(f"✗ Package import failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Error testing package installation: {e}")
        return False

def test_all_packages():
    """Test all packages"""
    print("Testing all packages...")
    
    packages = [
        "lightspeed-inline-agent",
        "lightspeed-agent", 
        "lightspeed-tool-runtime",
        "lightspeed-question-validity",
        "lightspeed-redaction"
    ]
    
    build_dir = Path("test_build")
    success_count = 0
    total_count = len(packages)
    
    for package_name in packages:
        print(f"\n--- Testing {package_name} ---")
        
        package_success = True
        
        # Test package structure
        if not test_package_structure(package_name, build_dir):
            package_success = False
        
        # Test pyproject.toml
        if not test_pyproject_toml(package_name, build_dir):
            package_success = False
        
        # Test package build
        if not test_package_build(package_name, build_dir):
            package_success = False
        
        # Test package installation
        if not test_package_install(package_name, build_dir):
            package_success = False
        
        if package_success:
            success_count += 1
            print(f"✓ {package_name} - ALL TESTS PASSED")
        else:
            print(f"✗ {package_name} - SOME TESTS FAILED")
    
    print(f"\n--- Test Summary ---")
    print(f"Success: {success_count}/{total_count} packages")
    
    return success_count == total_count

def cleanup():
    """Clean up test artifacts"""
    print("Cleaning up test artifacts...")
    
    # Remove test build directory
    test_build = Path("test_build")
    if test_build.exists():
        import shutil
        shutil.rmtree(test_build)
    
    # Uninstall test packages
    packages = [
        "lightspeed-inline-agent",
        "lightspeed-agent", 
        "lightspeed-tool-runtime",
        "lightspeed-question-validity",
        "lightspeed-redaction"
    ]
    
    for package in packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", package, "-y"
            ], capture_output=True)
        except:
            pass

def main():
    """Main test function"""
    print("Starting package build tests...")
    
    try:
        # Test build script
        if not test_build_script():
            print("✗ Build script test failed")
            return False
        
        # Test all packages
        if not test_all_packages():
            print("✗ Package tests failed")
            return False
        
        print("\n✓ ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    finally:
        cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 