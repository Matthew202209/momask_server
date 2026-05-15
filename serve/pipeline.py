import os
import torch
from os.path import join as pjoin
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models.utils.model_loader import load_vq_model, load_len_estimator, load_res_model, load_trans_model


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.vq_model = load_vq_model(config)
        self.res_model = load_res_model(config)
        self.t2m_transformer = load_trans_model(config)
        self.length_estimator = load_len_estimator(config)

        self.mean = np.load(pjoin(config.checkpoints_dir, config.dataset_name, config.vq_name, 'meta', 'mean.npy'))
        self.std = np.load(pjoin(config.checkpoints_dir, config.dataset_name, config.vq_name, 'meta', 'std.npy'))

        self._set_models()


    def _set_models(self):
        self.vq_model.eval()
        self.res_model.eval()
        self.t2m_transformer.eval()
        self.length_estimator.eval()
        self.vq_model.to(self.config.device)
        self.res_model.to(self.config.device)
        self.t2m_transformer.to(self.config.device)
        self.length_estimator.to(self.config.device)

    async def generate_motion(self, input):
        text_prompt = input['text_prompt']
        with torch.no_grad():
            token_lens = self._estimate_token_lens([text_prompt])
            m_length = token_lens * 4
            mids = self.t2m_transformer.generate([text_prompt], token_lens,
                                            timesteps=self.config.time_steps,
                                            cond_scale=self.config.cond_scale,
                                            temperature=self.config.temperature,
                                            topk_filter_thres=self.config.topkr,
                                            gsample=self.config.gumbel_sample)
            mids = self.res_model.generate(mids, [text_prompt], token_lens, temperature=1, cond_scale=5)
            pred_motions = self.vq_model.forward_decoder(mids)
            data = inv_transform(pred_motions)

            joint_data = data[0]
            joint_data = joint_data[:m_length[0]]
        return joint_data


    def _estimate_token_lens(self, text_prompt_list):
        text_embedding = self.t2m_transformer.encode_text(text_prompt_list)
        pred_dis = self.length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()
        return token_lens


async def inv_transform(self, data):
        return data * self.std + self.mean