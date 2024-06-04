import os
import time
from huggingface_hub import HfApi, CommitOperationAdd, create_branch


REPO_ID = "cloneofsimo/ye-pop-vae-t5-xl-metadata"
REPO_TYPE = "dataset"

# Initialize the API
api = HfApi()

api.upload_folder(
    folder_path='./laionmds_t5xl',
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)