import torch
import torch.nn as nn
from surface_uncertainty.model import Generate_center

# from visdom import Visdom
# viz = Visdom(server='http://127.0.0.1', port=8097)


class PAGNet_encoding(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = model_cfg.NUM_BEV_FEATURES

        self.generator = Generate_center(model_cfg.MODEL)
        self.generator.load_params_from_file_wo_logger(filename=model_cfg.CKPT, to_cpu=True)
        self.generator.cuda()
        self.generator.eval()

    def forward(self, batch_dict):
        topk_stable_idx_list = []

        batch_dict = self.generator.forward(batch_dict, training = False)
        
        # _, topk_stable_idx = torch.topk(-batch_dict['stds'], self.model_cfg.TOPK, dim=1)

        # topk_stable_idx_list.append(topk_stable_idx)

        return batch_dict




        