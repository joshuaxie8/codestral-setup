from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('mistral_models', 'Codestral-22B-v0.1') # download path
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Codestral-22B-v0.1", allow_patterns=["params.json", "config.json", "model-*.safetensors", "model.safetensors.index.json"], local_dir=mistral_models_path)