import os
from huggingface_hub import HfApi, Repository
import shutil

# Load environment variable for the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please configure it.")

# Model path
model_path = "focal_model.keras"

# Define the repository name
repo_name = "focal-loss-model"
repo_url = HfApi().create_repo(repo_name, token=hf_token, exist_ok=True)

# Clone or create the local repository
local_repo = Repository(local_dir=repo_name, clone_from=repo_url)

# Copy files into the repository directory
shutil.copy(model_path, f"{repo_name}/focal_model.keras")
with open(f"{repo_name}/README.md", "w") as readme:
    readme.write("# Focal Loss Model\n\nThis model is trained with Focal Loss.")

# Push changes to Hugging Face Hub
local_repo.push_to_hub()
print("Model successfully uploaded to Hugging Face Hub.")