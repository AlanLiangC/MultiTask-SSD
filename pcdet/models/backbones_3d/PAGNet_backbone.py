import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os
from ..backbones_2d.map_to_bev.projection import Projection
# from visdom import Visdom
# viz = Visdom(server='http://127.0.0.1', port=8097)


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
        self.conv3 = nn.Conv2d(inplanes, outplanes, 3, stride=stride, padding=1, bias=bias)
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


class PAGNet_Backbone(nn.Module):
    """ Backbone for PAGNet"""

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = kwargs['grid_size'][:2]

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []
        self.projs = []
        self.multi_bevs = nn.ModuleList()

        sa_config = self.model_cfg.SA_CONFIG
        self.bev_shape = sa_config.BEV_SHAPE
        self.multi_bevs_list = sa_config.MULTI_BEVS
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)


        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list): ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,                                                
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class = self.num_class
                    )
                )

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range
                                                                    )
                                       )

            channel_out_list.append(channel_out)

            self.projs.append(Projection(pc_range=model_cfg.POINT_CLOUD_RANGE, bev_shape=self.grid_size / self.bev_shape[k]))

            bev_layer_list = self.multi_bevs_list[k]
            self.multi_bevs.append(
                BasicBlock(inplanes=bev_layer_list[0], planes=bev_layer_list[1], outplanes=bev_layer_list[2], stride=bev_layer_list[-1])
            )


        self.num_point_features = channel_out


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        '''
        batch_dict:
            frame_id:       2
            gt_boxes:       torch.Size([2, 36, 8])
            points:         torch.Size([32768, 5])
            use_lead_xyz:   torch.Size([2])
            image_shape:    torch.Size([2, 2])
            batch_size:     2
        '''
        batch_size = batch_dict['batch_size']
        spatial_features_2d = batch_dict['spatial_features_2d']
        points = batch_dict['points']
        batch_idx, xyz, _ = self.break_up_pc(points)
        features = batch_dict['features']
        # vs_points = batch_dict['vs_points']

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        # li_cls_pred = None
        if batch_dict.get('li_cls_pred', None) is not None:
            li_cls_pred = batch_dict['li_cls_pred']
            li_cls_pred = li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1]).contiguous() 
            li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
            sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
        else:
            li_cls_pred = None


        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            elif self.layer_types[i] == 'Vote_Layer': #i=4
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                encoder_coords.append(torch.cat([center_origin_batch_idx[..., None].float(),centers_origin.view(batch_size, -1, 3)],dim =-1))
                    
            encoder_xyz.append(li_xyz)
            # vs_points.append(li_xyz.view(batch_size, -1, 3)[0,...])
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)        

            # if i > 0:    
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
            else:
                sa_ins_preds.append([])

            proj = self.projs[i]
            keep_bev = proj.init_bev_coord(encoder_coords[-1].view(-1, 4))[1]
            init_bev = proj.p2g_bev(encoder_features[-1].view(-1, li_features.shape[1])[keep_bev], batch_size)
            spatial_features_2d = torch.cat([spatial_features_2d, init_bev], dim = 1)
            spatial_features_2d = self.multi_bevs[i](spatial_features_2d)

            # viz.image(torch.where(torch.sum(init_bev[0], dim = 0)>0, 1, 0).float())


   
        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features
        batch_dict['spatial_features_2d'] = spatial_features_2d


        

        # save vs_points to txt

        # save_names = ['original_points','sample_points','DFPS1','DFPS2','ca1','ca2']
        # import numpy as np
        # for i in range(len(vs_points)):
        #     np.savetxt('../vspoints/kitti/{}.txt'.format(save_names[i]), vs_points[i].detach().cpu().numpy())
            
        
        
        ###save per frame 
        if self.model_cfg.SA_CONFIG.get('SAVE_SAMPLE_LIST',False) and not self.training:  
            import numpy as np 
            result_dir = np.load('/home/yifan/tmp.npy', allow_pickle=True)
            for i in range(batch_size)  :
                # i=0      
                # point_saved_path = '/home/yifan/tmp'
                point_saved_path = result_dir / 'sample_list_save'
                os.makedirs(point_saved_path, exist_ok=True)
                idx = batch_dict['frame_id'][i]
                xyz_list = []
                for sa_xyz in encoder_xyz:
                    xyz_list.append(sa_xyz[i].cpu().numpy()) 
                if '/' in idx: # Kitti_tracking
                    sample_xyz = point_saved_path / idx.split('/')[0] / ('sample_list_' + ('%s' % idx.split('/')[1]))

                    os.makedirs(point_saved_path / idx.split('/')[0], exist_ok=True)

                else:
                    sample_xyz = point_saved_path / ('sample_list_' + ('%s' % idx))

                np.save(str(sample_xyz), xyz_list)
                # np.save(str(new_file), point_new.detach().cpu().numpy())
        
        return batch_dict