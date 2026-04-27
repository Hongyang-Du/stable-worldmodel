import torch
from torch.nn import functional as F


def latent_loss(pred, target, loss_type: str):
    if loss_type == 'l1':
        return F.l1_loss(pred, target)
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    raise ValueError(f'Unsupported loss type: {loss_type}')


def strip_action_dims(tensor, action_range):
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def hwm_forward(self, batch, stage, cfg):
    """Forward/loss for a long-timescale HWM predictor.

    The first HWM training stage shares the PreJEPA/DINO-WM latent space and
    learns dynamics at a larger temporal stride. With PushT and
    ``frameskip=25``, the ``action`` input is a flattened 25-step primitive
    action block, serving as the high-level macro-action.
    """
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    prev_embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(prev_embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    loss_type = cfg.loss.type
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = latent_loss(
        pred_embedding[..., :pixels_dim],
        target_embedding[..., :pixels_dim],
        loss_type,
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == cfg.loss.action_key:
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = latent_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
                loss_type,
            )
        start = hi

    batch['actionless_emb'] = strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = strip_action_dims(
        prev_embedding, action_range
    )
    batch['actionless_pred_emb'] = strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = latent_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
        loss_type,
    )

    if batch['loss'].isnan():
        raise ValueError('NaN loss encountered!')

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch
