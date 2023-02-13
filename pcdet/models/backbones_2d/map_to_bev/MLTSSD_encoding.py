import torch
import torch.nn as nn
from ..unets import U_Net
from .projection import Projection




class MLTSSD_encoding(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.mlp_list = self.model_cfg.MLPS

        self.mlps = nn.ModuleList()
        self.shared_mlps = []
        bev_shape = grid_size[:2]
        input_channels = 4
        self.num_class = kwargs['num_class']
        self.npoint = model_cfg.NPOINT


        for dim in self.mlp_list:
            self.shared_mlps.extend([
                nn.Linear(input_channels, dim,bias=False),
                nn.ReLU()
            ])
            input_channels = dim

        self.mlps.append(nn.Sequential(*self.shared_mlps))
        self.proj = Projection(pc_range = model_cfg.POINT_CLOUD_RANGE, bev_shape = bev_shape)
        self.num_bev_features = self.mlp_list[-1] * 2
        self.encoder = U_Net(in_ch=self.mlp_list[-1], out_ch=self.mlp_list[-1])

        self.classifier = nn.Sequential(
            nn.Linear(self.num_bev_features,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16,self.num_class + 1)
        )


    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        coord = batch_dict['points'][:,:4]
        origin_pw_feature = batch_dict['points'][:,1:]
        assert origin_pw_feature.shape[-1] == 4
        for mlp in self.mlps:
            pw_feature = mlp(origin_pw_feature)
            origin_pw_feature = pw_feature

        keep_bev = self.proj.init_bev_coord(coord)[1]
        init_bev = self.proj.p2g_bev(pw_feature[keep_bev], batch_size)

        # U-Net to learning deeper features
        output_bev = self.encoder(init_bev)
        # new point-wise features
        pw_feature = self.proj.g2p_bev(output_bev)

        c_bev = pw_feature.shape[1]
        cmplt_pw_feature = output_bev.new_zeros([coord.shape[0], c_bev])
        cmplt_pw_feature[keep_bev] = pw_feature # Only change features in range
        cmplt_pw_feature = torch.cat([cmplt_pw_feature, origin_pw_feature], dim = -1)
        li_sem_pred = self.classifier(cmplt_pw_feature)

        new_points = []
        new_features = []
        for batch_idx in range(batch_size):
            batch_mask = coord[:,0] == batch_idx
            batch_points = batch_dict['points'][batch_mask]
            batch_features = cmplt_pw_feature[batch_mask]
            if batch_points.shape[0] <= self.npoint:
                emb_points = batch_points.new_zeros([self.npoint, batch_points.shape[-1]])
                emb_features = batch_points.new_zeros([self.npoint, batch_features.shape[-1]])
                emb_points[:,0] = batch_idx
                emb_points[:batch_points.shape[0],:] = batch_points
                emb_features[:batch_points.shape[0],:] = batch_features
                new_points.append(emb_points)
                new_features.append(emb_features)
            else:
                batch_sem_pred = li_sem_pred[batch_mask][:,1:]
                cls_features_max, class_pred = batch_sem_pred.max(dim=-1)
                score_pred = torch.sigmoid(cls_features_max) # B,N
                score_picked, sample_idx = torch.topk(score_pred, self.npoint, dim=-1) 

                new_points.append(batch_points[sample_idx])
                new_features.append(batch_features[sample_idx])

        points = torch.cat(new_points, dim = 0)
        new_features = torch.cat(new_features, dim = 0)

        batch_dict.update({
            'features': new_features,
            'points': points,
            'sem_pred': li_sem_pred
        })
        

        # batch_dict['li_cls_pred'] = li_cls_pred

        return batch_dict




        