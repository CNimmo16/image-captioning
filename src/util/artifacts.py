import torch
import pandas as pd
import os
import wandb

import os
dirname = os.path.dirname(__file__)

api = wandb.Api()

def download_from_wandb(ref: str, file: str, type: str, meta: dict = None):
    env_override = os.environ.get(f"{ref.upper().replace('-', '_')}_PATH", None)
    if env_override:
        print(f"INFO: Using override for {ref}: {env_override}")
        return env_override

    if meta:
        artifact_collections = api.artifact_type(type_name=type, project='cnimmo16/image-captioning').collections()
        artifact = None
        for coll in artifact_collections:
            if coll.name == ref:
                for a in coll.artifacts():
                    if (a.metadata == meta):
                        artifact = a
                        print(f"Found version {a.version} of {ref}")
                        break
                else:
                    continue  # only executed if the inner loop did NOT break
                break
        if artifact is None:
            raise Exception(f"Could not find artifact {ref} with metadata {meta}")
    else:
        artifact = api.artifact(f"cnimmo16/image-captioning/{ref}:latest")
    directory = artifact.download(os.path.join(dirname, '../../artifacts'))
    return os.path.join(directory, file)
    
def load_artifact(ref: str, type: str, meta: dict = None):
    if (type == 'model'):
        weights_path = download_from_wandb(ref, f"{ref}.generated.pt", type, meta)
        return torch.load(weights_path, map_location=torch.device('cpu'))

    raise Exception(f'Unknown artifact: {ref}')

def store_artifact(ref: str, type: str, file: str, meta: dict):
    artifact = wandb.Artifact(ref, type=type, metadata=meta)
    artifact.add_file(file)
    wandb.log_artifact(artifact)
