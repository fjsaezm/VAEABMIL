import torch

from omegaconf import OmegaConf


class MLP(torch.nn.Module):
    def __init__(self, in_dim, dim=512, n_layers=1):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, dim)
        self.fc_layers = torch.nn.Sequential(
            *[torch.nn.Linear(dim, dim) for _ in range(n_layers - 1)]
        )
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if len(self.fc_layers) > 0:
            for layer in self.fc_layers:
                x = layer(x)
                x = self.act(x)
        return x


def build_model(config, in_dim, pos_weight=None):

    ce_criterion = torch.nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
    feat_ext = MLP(in_dim, 512, 1)
    params_dict = OmegaConf.to_container(
        config.model.params, resolve=True, throw_on_missing=True
    )
    print("Model params:", params_dict)

    if config.model.name == "abmil":
        from torchmil.models import ABMIL

        return ABMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "sm_abmil":
        from torchmil.models import SmABMIL

        return SmABMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "prob_smooth_abmil":
        from torchmil.models import ProbSmoothABMIL

        return ProbSmoothABMIL(
            in_shape=(in_dim,),
            feat_ext=feat_ext,
            criterion=ce_criterion,
            **{k: v for k, v in params_dict.items() if k != "annealing"},
        )
    elif config.model.name == "clam":
        from torchmil.models import CLAM_SB

        return CLAM_SB(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "dsmil":
        from torchmil.models import DSMIL

        return DSMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "dtfdmil":
        from torchmil.models import DTFDMIL

        return DTFDMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "patchgcn":
        from torchmil.models import PatchGCN

        return PatchGCN(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "deepgraphsurv":
        from torchmil.models import DeepGraphSurv

        return DeepGraphSurv(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "transformer_abmil":
        from torchmil.models import TransformerABMIL

        return TransformerABMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "sm_transformer_abmil":
        from torchmil.models import SmTransformerABMIL

        return SmTransformerABMIL(
            in_shape=(in_dim,), feat_ext=feat_ext, criterion=ce_criterion, **params_dict
        )
    elif config.model.name == "transformer_prob_smooth_abmil":
        from torchmil.models import TransformerProbSmoothABMIL

        return TransformerProbSmoothABMIL(
            in_shape=(in_dim,),
            feat_ext=feat_ext,
            criterion=ce_criterion,
            **{k: v for k, v in params_dict.items() if k != "annealing"},
        )
    elif config.model.name == "transmil":
        from torchmil.models import TransMIL

        return TransMIL(
            in_shape=(in_dim,), criterion=ce_criterion, feat_ext=feat_ext, **params_dict
        )
    elif config.model.name == "camil":
        from torchmil.models import CAMIL

        return CAMIL(
            in_shape=(in_dim,), criterion=ce_criterion, feat_ext=feat_ext, **params_dict
        )
    elif config.model.name == "iibmil":
        from torchmil.models import IIBMIL

        return IIBMIL(
            in_shape=(in_dim,), criterion=ce_criterion, feat_ext=feat_ext, **params_dict
        )
    elif config.model.name == "setmil":
        from torchmil.models import SETMIL

        return SETMIL(
            in_shape=(in_dim,), criterion=ce_criterion, feat_ext=feat_ext, **params_dict
        )
    elif config.model.name == "gtp":
        from torchmil.models import GTP

        return GTP(
            in_shape=(in_dim,), criterion=ce_criterion, feat_ext=feat_ext, **params_dict
        )
    elif config.model.name == "vaeabmil":
        from models.VAEABMIL import VAEABMIL
        
        return VAEABMIL(
            in_shape=(in_dim,),
            criterion=ce_criterion,
            **params_dict
        )
    else:
        raise NotImplementedError(f"Model {config.model.name} not implemented")


def build_autoencoder(config, in_dim):
    """Build an autoencoder model."""
    from models.nn.VariationalAutoEncoder import VariationalAutoEncoderMIL
    import argparse  # Needed because of the implementation of the autoencoder

    config["model_config"] = vars(
        eval(config["model_config"], {"Namespace": argparse.Namespace})
    )
    ae_params = OmegaConf.create(config)
    return VariationalAutoEncoderMIL(
        input_shape=(in_dim,),
        model_name=ae_params.model_config.architecture,
        covar_mode="single",
    )
