import torch
import numpy as np
import torch.nn as nn
from ..unets import U_Net
from .projection import Projection


class Classifier(nn.Module):
    def __init__(self, input_channels, layers, sem_class):
        super(Classifier, self).__init__()
        self.later_list = layers
        self.act = nn.ReLU()
        self.shared_mlps = []
        self.input_channels = input_channels
        for dim in self.later_list:
            self.shared_mlps.extend([
            nn.Linear(input_channels, dim,bias=False),
            nn.ReLU(),
            nn.Dropout(0.2)
            ])
            input_channels = dim
        self.shared_mlps.extend([
            nn.Linear(dim, sem_class, bias=False)
            ])
        self.classifier = nn.Sequential(*self.shared_mlps)

    def forward(self, input_features):
        assert input_features.shape[-1] == self.input_channels
        return self.classifier(input_features)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, outplanes, stride=1, norm_fn=nn.BatchNorm2d, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=1, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, inplanes, 3, stride=1, padding=1, bias=bias)
        self.bn2 = norm_fn(inplanes)
        self.conv3 = nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=bias)
        self.bn3 = norm_fn(outplanes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return self.relu(self.bn3(self.conv3(out)))


class PAGNet_encoding(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.mlp_list = self.model_cfg.MLPS

        self.mlps = nn.ModuleList()
        self.shared_mlps = []
        
        input_channels = 4
        # self.num_class = kwargs['num_class']
        self.sem_num_class = model_cfg.SEM_CLASS_NUM
        self.npoint = model_cfg.NPOINT
        self.point_cloud_range = np.array(self.model_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(model_cfg.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        bev_shape = self.grid_size[:2]


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

        self.classifier = Classifier(input_channels=self.num_bev_features, layers=model_cfg.CLASSIFIER, sem_class=self.sem_num_class)
        self.sample_feature_ln = BasicBlock(inplanes=self.mlp_list[-1]*2, planes=self.mlp_list[-1]*2, outplanes=self.mlp_list[-1]*4)

        # self.norm_feature_layers = nn.Sequential(
        #     nn.Conv2d(self.mlp_list[-1]*3, self.mlp_list[-1]*2, 3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(),
        #     nn.ReLU()
        # )

 
    def cosine_similarity(self,x,y):
        num = torch.matmul(x,y.T).squeeze()
        denom = torch.norm(x, dim = -1) * torch.norm(y.squeeze())
        return num / denom


    def forward(self, batch_dict):
        # visible points
        # vs_points = []

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

        # kitti
        new_points = []
        new_features = []
        for batch_idx in range(batch_size):
            batch_mask = coord[:,0] == batch_idx
            batch_points = batch_dict['points'][batch_mask]
            batch_features = cmplt_pw_feature[batch_mask]
            # if batch_idx == 0:
                # vs_points.append(torch.cat([batch_points[:,1:4],batch_dict['fake_labels'][batch_mask].view(-1,1)], dim = -1))

            if batch_points.shape[0] <= self.npoint:
                emb_points = batch_points.new_zeros([self.npoint, batch_points.shape[-1]])
                emb_features = batch_points.new_zeros([self.npoint, batch_features.shape[-1]])
                emb_points[:,0] = batch_idx
                emb_points[:batch_points.shape[0],:] = batch_points
                emb_features[:batch_points.shape[0],:] = batch_features
                new_points.append(emb_points)
                new_features.append(emb_features)
            else:
                batch_sem_pred = li_sem_pred[batch_mask]
                batch_sem_args = torch.argmax(li_sem_pred[batch_mask], dim = -1)
                fg_tag = batch_sem_args > 0
                if torch.sum(fg_tag) >= self.npoint:
                    batch_points = batch_points[fg_tag]
                    batch_features = batch_features[fg_tag]
                    batch_sem_pred = batch_sem_pred[fg_tag][:,1:]
                    cls_features_max, class_pred = batch_sem_pred.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    _, sample_idx = torch.topk(score_pred, self.npoint, dim=-1) 
                    new_points.append(batch_points[sample_idx])
                    new_features.append(batch_features[sample_idx])
                else:
                    last_npoint = self.npoint - torch.sum(fg_tag)
                    fg_points = batch_points[fg_tag]
                    fp_features = batch_features[fg_tag]
                    
                    bg_points = batch_points[~fg_tag]
                    bg_features = batch_features[~fg_tag]
                    batch_bg_sem_pred = batch_sem_pred[~fg_tag]

                    abs_bg = batch_points.new_zeros([1,self.sem_num_class])
                    abs_bg[0,0] = 1
                    abs_cos_features = self.cosine_similarity(torch.sigmoid(batch_bg_sem_pred), abs_bg)
                    _, sample_idx = torch.topk(-abs_cos_features, last_npoint, dim=-1) 
                    soft_bg_points = bg_points[sample_idx]
                    soft_bg_features = bg_features[sample_idx]

                    batch_points = torch.cat([fg_points, soft_bg_points], dim = 0)
                    batch_features = torch.cat([fp_features, soft_bg_features], dim = 0)

                    assert batch_points.shape[0] == self.npoint
                    new_points.append(batch_points)
                    new_features.append(batch_features)

        # vs_points.append(new_points[0][:,1:4])
        points = torch.cat(new_points, dim = 0)
        new_features = torch.cat(new_features, dim = 0)


        batch_dict.update({
            'features': new_features,
            'points': points,
            'sem_pred': li_sem_pred,
            # 'vs_points': vs_points
        })

        new_coord = points[:,:4]
        new_keep_bev = self.proj.init_bev_coord(new_coord)[1]
        new_bev_feature = self.proj.p2g_bev(new_features[new_keep_bev], batch_size) 

        new_bev_feature[:,:self.mlp_list[-1],...] += output_bev


        spatial_features_2d = new_bev_feature
        spatial_features_2d = self.sample_feature_ln(spatial_features_2d)

        batch_dict.update({
            'grid_size': self.grid_size,
            'spatial_features_2d': spatial_features_2d
        })

        
        

        # batch_dict['li_cls_pred'] = li_cls_pred

        return batch_dict




        