from omegaconf import OmegaConf
import json
import wandb
from .models import build_model, build_autoencoder
import os
import torch

PATH_CONFIGS = "/work/work_fjaviersaezm/lvaer-mil/code/configs/dataset/"
BASE_PATH_RUNS = "/work/work_fjaviersaezm/benchmark-mil/wandb/"
BASE_VAE_RUNS = "/work/work_fjaviersaezm/StandaloneOOD/wandb/"


def get_in_dim(dataset):
    X = dataset[0]["X"]
    return X.shape[1]


def load_autoencoder(config, wandb_api, in_dim):

    run = wandb_api.run(
        f"{config.wandb.user}/{config.wandb.vae_project}/{config.dataset.vae_run}"
    )
    run_cfg = OmegaConf.create(dict(run.config))
    vae = build_autoencoder(run_cfg, in_dim=in_dim)
    vae_folder = [f for f in os.listdir(BASE_VAE_RUNS) if f.endswith(run.id)][0]
    vae_weights_path = os.path.join(BASE_VAE_RUNS, vae_folder, "files/weights/best.pt")
    print(f"Loading VAE weights from {vae_weights_path}")
    vae.load_state_dict(torch.load(vae_weights_path))
    vae.eval()
    return vae


def load_threshold(config):
    model_wandb_run = config.dataset.vae_run
    with open(f"{config.thresholds_dir}/{model_wandb_run}.json", "r") as f:
        thresholds = json.load(f)

    if (
        config.dataset.name != thresholds["dataset_name"]
        or config.dataset.features_dir != thresholds["features"]
    ):
        raise ValueError(
            f"Dataset mismatch: {config.dataset.name} != {thresholds['dataset_name']} or {config.dataset.features_dir} != {thresholds['features']}"
        )
    return thresholds["thresholds"][config.ood_score][str(config.percentile)]


def load_mil_model(run, in_dim):
    run_cfg = OmegaConf.create(dict(run.config))
    run_folder = [f for f in os.listdir(BASE_PATH_RUNS) if f.endswith(run.id)][0]
    print(f"Testing model {run_cfg.model.name} and ID {run.id}...")
    model = build_model(run_cfg, in_dim, pos_weight=None)
    # Load the model weights
    model.load_state_dict(
        torch.load(BASE_PATH_RUNS + run_folder + "/files/weights/best.pt")
    )
    model.eval()
    return model


def create_or_resume_run(run_id, run_path, project, entity=None):
    # Try to resume if it exists, else create
    try:
        api = wandb.Api()
        _ = api.run(run_path)
        resume_mode = "must"
    except wandb.errors.CommError:
        resume_mode = "never"

    run = wandb.init(project=project, entity=entity, id=run_id, resume=resume_mode)
    return run
