import os
import torch
import logging
from os.path import join as pjoin
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
    ckpt = torch.load(pjoin(config.checkpoints_dir, config.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=config.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model
