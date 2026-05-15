from pydantic import BaseModel
from ray.serve import Application

from serve.deployment import MotionGenerator

class VAEConfig(BaseModel):
    dim_pose: int
    nb_code: int
    code_dim: int
    output_emb_width: int
    down_t: int
    stride_t: int
    width: int
    depth: int
    dilation_growth_rate: int
    vq_act: int
    vq_norm: int


class TransModelConfig(BaseModel):
    code_dim: int
    latent_dim: int
    clip_dim: int
    ff_size: int
    n_layers: int
    n_heads: int
    dropout: int
    cond_drop_prob: int


class ResModelConfig(BaseModel):
    num_quantizers: int
    num_tokens: int
    shared_codebook: int
    cond_drop_prob: float
    share_weight: bool




class MomaskArgs(VAEConfig):


class MomaskArgs(BaseModel):
    pass


def build_app(args: MomaskArgs) -> Application:
    return MotionGenerator.bind(args)


motion_generator_app = build_app(MomaskArgs())