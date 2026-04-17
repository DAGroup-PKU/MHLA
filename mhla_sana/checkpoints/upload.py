from huggingface_hub import HfApi

api = HfApi()

# Upload a single file
api.upload_file(
    path_or_fileobj="./SANA_MHLA.pth",
    path_in_repo="SANA_MHLA/SANA_MHLA.pth",
    repo_id="DAGroup-PKU/MHLA", 
    repo_type="model"
)
