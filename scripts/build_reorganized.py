#!/usr/bin/env python3
"""
Build script for the reorganized provider packages.

This script builds all provider packages from the new organized structure.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Provider definitions with their new locations
PROVIDERS = {
    "lightspeed-inline-agent": {
        "path": "providers/agents/lightspeed-inline-agent",
        "description": "Lightspeed inline agent provider for llama-stack",
        "dependencies": ["llama-stack==0.2.16", "httpx", "pydantic>=2.10.6"],
    },
    "lightspeed-agent": {
        "path": "providers/agents/lightspeed-agent",
        "description": "Lightspeed remote agent provider for llama-stack",
        "dependencies": ["llama-stack==0.2.16", "httpx", "pydantic>=2.10.6"],
    },
    "lightspeed-question-validity": {
        "path": "providers/safety/lightspeed-question-validity",
        "description": "Lightspeed question validity safety provider for llama-stack",
        "dependencies": ["llama-stack==0.2.16", "httpx", "pydantic>=2.10.6"],
    },
    "lightspeed-redaction": {
        "path": "providers/safety/lightspeed-redaction",
        "description": "Lightspeed redaction safety provider for llama-stack",
        "dependencies": ["llama-stack==0.2.16", "httpx", "pydantic>=2.10.6"],
    },
    "lightspeed-tool-runtime": {
        "path": "providers/tool-runtime/lightspeed-tool-runtime",
        "description": "Lightspeed tool runtime provider for llama-stack",
        "dependencies": ["llama-stack==0.2.16", "httpx", "pydantic>=2.10.6", "mcp"],
    },
}


def get_version() -> str:
    """Get version from the main pyproject.toml"""
    import tomllib
    
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]


def build_package(provider_name: str, provider_config: Dict, version: str) -> bool:
    """Build a single package"""
    package_path = Path(provider_config["path"])
    
    if not package_path.exists():
        print(f"Error: Package path does not exist: {package_path}")
        return False
    
    try:
        # Change to package directory
        original_dir = os.getcwd()
        os.chdir(package_path)
        
        # Update version in pyproject.toml
        update_version_in_pyproject(version)
        
        # Build the package
        result = subprocess.run(
            [sys.executable, "-m", "build"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error building {provider_name}: {result.stderr}")
            return False
        
        print(f"Successfully built {provider_name}")
        return True
        
    except Exception as e:
        print(f"Error building {provider_name}: {e}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)


def update_version_in_pyproject(version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    
    if not pyproject_path.exists():
        return
    
    # Read current content
    with open(pyproject_path, "r") as f:
        content = f.read()
    
    # Update version
    import re
    content = re.sub(r'version = ".*?"', f'version = "{version}"', content)
    
    # Write back
    with open(pyproject_path, "w") as f:
        f.write(content)


def main():
    """Main build function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build reorganized provider packages")
    parser.add_argument("--provider", help="Build specific provider only")
    parser.add_argument("--version", help="Override version (defaults to main pyproject.toml)")
    
    args = parser.parse_args()
    
    # Get version
    version = args.version or get_version()
    
    # Determine which providers to build
    providers_to_build = [args.provider] if args.provider else PROVIDERS.keys()
    
    if args.provider and args.provider not in PROVIDERS:
        print(f"Unknown provider: {args.provider}")
        print(f"Available providers: {list(PROVIDERS.keys())}")
        sys.exit(1)
    
    success_count = 0
    total_count = len(providers_to_build)
    
    for provider_name in providers_to_build:
        if provider_name not in PROVIDERS:
            print(f"Skipping unknown provider: {provider_name}")
            continue
            
        print(f"\nBuilding {provider_name}...")
        
        provider_config = PROVIDERS[provider_name]
        
        # Build the package
        if build_package(provider_name, provider_config, version):
            success_count += 1
    
    print(f"\nBuild complete: {success_count}/{total_count} packages built successfully")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main() 