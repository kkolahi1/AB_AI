#!/usr/bin/env python3

"""
Script to check installed dependencies for troubleshooting
"""

import torch
import sys
import pkg_resources
import importlib
import platform

def check_imports():
    """Check if critical imports work"""
    success = True
    critical_imports = [
        "transformers",
        "torch", 
        "streamlit",
        "langchain_core",
        "langchain_openai",
        "langgraph",
        "numpy",
        "scipy"
    ]

    print("\n=== Import Checks ===")
    for module_name in critical_imports:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Unknown")
            print(f"✓ {module_name}: {version}")
        except ImportError as e:
            success = False
            print(f"✗ {module_name}: FAILED - {str(e)}")
    return success

def check_transformers_model():
    """Check if transformers can load your fine-tuned model"""
    print("\n=== Transformers Model Check ===")
    try:
        from transformers import AutoModel, AutoTokenizer
        model_name = "kamkol/ab_testing_finetuned_arctic_ft-36dfff22-0696-40d2-b3bf-268fe2ff2aec"
        print(f"Testing loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        inputs = tokenizer("Test sentence", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print("✓ Successfully loaded model and ran forward pass")
        return True
    except Exception as e:
        print(f"✗ Transformers model test failed: {str(e)}")
        return False

def check_langchain_components():
    """Check critical LangChain components"""
    print("\n=== LangChain Check ===")
    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph
        print("✓ LangChain core components imported successfully")
        return True
    except Exception as e:
        print(f"✗ LangChain check failed: {str(e)}")
        return False

def print_system_info():
    """Print system information"""
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Implementation: {platform.python_implementation()}")

def main():
    """Main function to run checks"""
    print("=== Dependency Check ===")
    print_system_info()

    # Get all installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    # Check for specific dependencies
    key_packages = [
        "transformers",
        "torch",
        "streamlit",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "numpy",
        "scipy"
    ]

    print("\n=== Package Versions ===")
    for pkg in key_packages:
        version = installed_packages.get(pkg, "Not installed")
        print(f"{pkg}: {version}")

    # Test imports
    imports_ok = check_imports()
    
    # Test transformers model
    tf_ok = check_transformers_model()
    
    # Test LangChain
    lc_ok = check_langchain_components()

    # Final result
    if imports_ok and tf_ok and lc_ok:
        print("\n✓ All critical checks passed")
    else:
        print("\n✗ Some checks failed, see details above")

if __name__ == "__main__":
    main()
