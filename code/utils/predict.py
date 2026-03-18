import numpy as np
import torch

from tqdm import tqdm

from torchmil.nn import masked_softmax
from torchmil.models import MILModelWrapper


def predict(model, dataloader, device="cuda", desc="Test"):

    # model = MILModelWrapper(model).to(device)
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description(desc)

    bag_idx_list = []
    Y_list = []
    Y_logits_pred_list = []
    y_list = []
    f_pred_list = []
    s_pred_list = []

    for bag_idx, batch in pbar:
        batch_size = batch["Y"].size(0)
        if batch_size != 1:
            raise ValueError("Batch size must be 1")
        # batch.pop("mask")

        batch = batch.to(device)

        Y = batch["Y"]  # (batch_size, 1)
        y = batch["y_inst"]  # (batch_size, bag_size)

        Y_logits_pred = model.predict(
            batch, return_inst_pred=False
        )  # Y_logits_pred: (batch_size, 1)
        f_pred = torch.rand(batch_size, batch["X"].size(1), 1).to(
            device
        )  # f_pred: (batch_size, bag_size, 1)

        # Y_logits_pred, f_pred = model.predict(batch, return_inst_pred=True) # Y_logits_pred: (batch_size,), f_pred: (batch_size, bag_size)

        Y_logits_pred = Y_logits_pred.detach()  # (batch_size,)
        f_pred = f_pred.detach()  # (batch_size, bag_size)

        if f_pred.dim() == 3:
            f_pred = f_pred.squeeze(-1)

        if Y_logits_pred.dim() == 3:
            Y_logits_pred = Y_logits_pred.squeeze(-1)

        s_pred = masked_softmax(f_pred)  # (batch_size, bag_size)

        f_pred = f_pred.view(-1)  # (batch_size*bag_size,)
        s_pred = s_pred.view(-1)  # (batch_size*bag_size,)
        y = y.view(-1)

        Y_list.append(Y.cpu().numpy())
        Y_logits_pred_list.append(Y_logits_pred.cpu().numpy())
        y_list.append(y.cpu().numpy())
        f_pred_list.append(f_pred.cpu().numpy())
        s_pred_list.append(s_pred.cpu().numpy())
        bag_idx_list.append(np.repeat(bag_idx, len(y)))

    Y = np.concatenate(Y_list)  # (batch_size*bag_size,)
    y = np.concatenate(y_list, axis=0)  # (batch_size*bag_size, 1)

    Y_logits_pred = np.concatenate(Y_logits_pred_list)  # (batch_size,)
    f_pred = np.concatenate(f_pred_list, axis=0)  # (batch_size, 1)
    s_pred = np.concatenate(s_pred_list, axis=0)  # (batch_size, 1)
    bag_idx = np.concatenate(bag_idx_list, axis=0)  # (batch_size*bag_size,)

    # Discard unlabeled instances
    keep_idx = np.where((y == 0) | (y == 1))[0]
    y = y[keep_idx]
    f_pred = f_pred[keep_idx]
    s_pred = s_pred[keep_idx]
    bag_idx = bag_idx[keep_idx]

    return Y, y, Y_logits_pred, f_pred, s_pred, bag_idx
