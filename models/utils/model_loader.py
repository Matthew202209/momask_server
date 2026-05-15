import os
import torch
import logging
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

logger = logging.getLogger("ray.serve")


def load_vq_model(config):
    vq_model = RVQVAE(config,
                config.dim_pose,
                config.nb_code,
                config.code_dim,
                config.output_emb_width,
                config.down_t,
                config.stride_t,
                config.width,
                config.depth,
                config.dilation_growth_rate,
                config.vq_act,
                config.vq_norm)

    # 路径拼接
    ckpt_path = pjoin(config.checkpoints_dir, config.dataset_name, config.name, 'model', 'net_best_fid.tar')

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])

    # 打印加载完成日志
    logger.info(f"Loading VQ Model Completed! Path: {ckpt_path}")

    return vq_model


def load_len_estimator(config):
    model = LengthEstimator(512, 50)
    ckpt_path = pjoin(config.checkpoints_dir, config.dataset_name, 'length_estimator', 'model', 'finest.tar')
    ckpt = torch.load(ckpt_path,
                      map_location=config.device)
    model.load_state_dict(ckpt['estimator'])
    logger.info(f"Loading Length Estimator Completed! Path: {ckpt_path}")
    return model


def load_trans_model(config):
    clip_version = 'ViT-B/32'
    t2m_transformer = MaskTransformer(code_dim=config.code_dim,
                                      cond_mode='text',
                                      latent_dim=config.latent_dim,
                                      ff_size=config.ff_size,
                                      num_layers=config.n_layers,
                                      num_heads=config.n_heads,
                                      dropout=config.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=config.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=config)

    ckpt_path = pjoin(config.checkpoints_dir, config.dataset_name, config.name, 'model', 'latest.tar')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    logger.info(f"Loading Transformer Completed! Path: {ckpt_path}")
    return t2m_transformer


def load_res_model(config):
    clip_version = 'ViT-B/32'
    res_transformer = ResidualTransformer(code_dim=config.code_dim,
                                            cond_mode='text',
                                            latent_dim=config.latent_dim,
                                            ff_size=config.ff_size,
                                            num_layers=config.n_layers,
                                            num_heads=config.n_heads,
                                            dropout=config.dropout,
                                            clip_dim=512,
                                            shared_codebook=config.shared_codebook,
                                            cond_drop_prob=config.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=config.share_weight,
                                            clip_version=clip_version,
                                            opt=config)

    ckpt_path = pjoin(config.checkpoints_dir, config.dataset_name, config.name, 'model', 'net_best_fid.tar')
    ckpt = torch.load(ckpt_path,
                      map_location=config.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    logger.info(f"Loading Residual Transformer  Completed! Path: {ckpt_path}")
    return res_transformer