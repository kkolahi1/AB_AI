#!/usr/bin/env python
import os
import argparse
import subprocess
from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder

# Load environment variables
load_dotenv()

# Get Huggingface token from environment or arguments
HF_TOKEN = os.getenv("HF_TOKEN")

def process_data():
    """Run the data processing script"""
    print("Processing PDF data...")
    subprocess.run(["python", "process_data.py"], check=True)
    print("Data processing complete.")

def validate_processed_data():
    """Validate that processed data exists"""
    data_path = "data/processed_data"
    if not os.path.exists(data_path):
        raise ValueError(f"Processed data directory {data_path} not found. Run 'python process_data.py' first.")
    
    # Check for chunks.pkl
    if not os.path.exists(os.path.join(data_path, "chunks.pkl")):
        raise ValueError("chunks.pkl not found in processed data directory.")
    
    # Check for Qdrant collection files
    collection_files = [f for f in os.listdir(data_path) if f.startswith("kohavi_ab_testing_pdf_collection")]
    if not collection_files:
        raise ValueError("Qdrant collection files not found in processed data directory.")
    
    print("✅ Processed data validation passed")

def build_docker_image(tag="ab-testing-qa:latest"):
    """Build the Docker image"""
    print(f"Building Docker image: {tag}")
    subprocess.run(["docker", "build", "-t", tag, "."], check=True)
    print("Docker image built successfully.")

def upload_processed_data(hf_token=None, space_name=None):
    """Upload processed data to Hugging Face persistent storage"""
    api = HfApi(token=hf_token or HF_TOKEN)
    
    upload_folder(
        folder_path="data/processed_data",
        repo_id=space_name,
        repo_type="space",
        path_in_repo="data/processed_data"
    )
    print("✅ Uploaded processed data to Hugging Face Spaces")

def push_to_huggingface(hf_token=None, space_name=None):
    """Push Docker image to Huggingface"""
    upload_processed_data(hf_token, space_name)

    if not hf_token and not HF_TOKEN:
        raise ValueError("Huggingface token not provided. Either set HF_TOKEN environment variable or pass with --token")
    
    token = hf_token or HF_TOKEN
    
    if not space_name:
        raise ValueError("Huggingface space name not provided. Use --space parameter")
    
    print(f"Logging in to Huggingface container registry...")
    login_cmd = ["docker", "login", "-u", "user", "--password", token, "registry.hf.space"]
    subprocess.run(login_cmd, check=True)
    
    # Tag the image for Huggingface
    hf_tag = f"registry.hf.space/{space_name}/ab-testing-qa:latest"
    subprocess.run(["docker", "tag", "ab-testing-qa:latest", hf_tag], check=True)
    
    # Push the image
    print(f"Pushing Docker image to {hf_tag}...")
    subprocess.run(["docker", "push", hf_tag], check=True)
    
    print(f"✅ Successfully pushed to Huggingface space: {space_name}")
    print(f"Visit your space at: https://huggingface.co/spaces/{space_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy AB Testing QA App to Huggingface")
    parser.add_argument("--process", action="store_true", help="Process PDF data")
    parser.add_argument("--validate", action="store_true", help="Validate processed data")
    parser.add_argument("--build", action="store_true", help="Build Docker image")
    parser.add_argument("--push", action="store_true", help="Push to Huggingface")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--token", help="Huggingface token (if not in .env)")
    parser.add_argument("--space", help="Huggingface space name (username/space)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.all or args.process:
        process_data()
    
    if args.all or args.validate:
        validate_processed_data()
    
    if args.all or args.build:
        build_docker_image()
    
    if args.all or args.push:
        push_to_huggingface(args.token, args.space)

if __name__ == "__main__":
    main() 