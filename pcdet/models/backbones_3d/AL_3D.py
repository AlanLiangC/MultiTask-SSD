import torch 
import torch.nn as nn
# from ..backbones_2d.cpgnet_moudles import Projection
from ..backbones_2d.AL_2D import CP_Unet
from ..backbones_2d.map_to_bev.projection import Projection
from .MLTSSD_backbone import MLTSSD_Backbone
# from visdom import Visdom
# viz = Visdom(server='http://127.0.0.1', port=8097)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Space2Depth(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super().__init__()
        self.compress_conv = nn.Sequential(nn.Conv2d(input_channels, output_channels,kernel_size=(1,1)),
                                            nn.BatchNorm2d(output_channels),
                                            nn.ReLU())

    def forward(self, x, down_scale):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, down_scale, stride=down_scale)
        out = unfolded_x.view(n, c * down_scale ** 2, h // down_scale, w // down_scale)
        out = self.compress_conv(out)
        return out



class FusionBlock(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.cbam1 = CBAM(inplanes=input_channels,planes=input_channels)
        self.cbam2 = nn.Sequential(CBAM(inplanes=input_channels,planes=input_channels),
                                    nn.Conv2d(input_channels,input_channels // 2,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(input_channels // 2),
                                    nn.ReLU())
        self.cbam3 = nn.Sequential(CBAM(inplanes=input_channels // 2,planes=input_channels //2),
                                    nn.Conv2d(input_channels // 2,input_channels // 4,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(input_channels // 4),
                                    nn.ReLU())

        self.transconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels,
                                            out_channels=input_channels // 2,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                        nn.BatchNorm2d(input_channels // 2),
                                        nn.ReLU())

        self.transconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels // 2,
                                            out_channels=input_channels // 4,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                        nn.BatchNorm2d(input_channels // 4),
                                        nn.ReLU())

        self.transconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels // 4,
                                            out_channels=input_channels // 8,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                            nn.BatchNorm2d(input_channels // 8),
                                            nn.ReLU())

        self.sd1 = Space2Depth(input_channels // 2, input_channels // 4)
        self.sd2 = Space2Depth(input_channels, input_channels // 2)
        self.sd3 = Space2Depth(input_channels // 2, input_channels // 2)
        
        

    def forward(self, encode_dict, proj):
        coord = encode_dict['coord']
        e1 = encode_dict['e3'] # [2, 128, 4, 256]
        e2 = encode_dict['e2'] # [2, 64, 8, 512]
        e3 = encode_dict['e1'] # [2, 32, 16, 1024]

        batch_size = e1.shape[0]

        e1 = self.cbam1(e1) # [2, 128, 4, 256]
        e1 = self.transconv1(e1) # [2, 64, 8, 512]

        e2 = torch.cat([e1, e2], dim = 1) # [2, 128, 8, 512]
        e2 = self.cbam2(e2) # [2, 64, 8, 512]
        e2 = self.transconv2(e2) # [2, 32, 16, 1024]

        e3 = torch.cat([e2, e3], dim = 1)
        e3 = self.cbam3(e3)
        e3 = self.transconv3(e3) # [2, 16, 32, 2048]
    
        # project RV-->PW
        keep_range = proj.init_range_coord(coord)[1]
        range2pw = proj.g2p_range(e3)
        c_range = e3.shape[1]
        cmplt_range2pw = range2pw.new_zeros([coord.shape[0], c_range])
        cmplt_range2pw[keep_range] = range2pw

        # PW --> BEV
        keep_bev = proj.init_bev_coord(coord)[1]
        pw2bev = proj.p2g_bev(cmplt_range2pw[keep_bev], batch_size)

        # Space2Depth + Conv 1x1
        sd1 = self.sd1(pw2bev, down_scale = 2) # [2, 32, 304, 304]
        sd2 = self.sd2(sd1, down_scale = 2) # [2, 64, 152, 152]
        sd3 = self.sd3(sd2, down_scale = 1) # [2, 64, 152, 152]

        return sd3


class FusionBlockV2(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.cbam1 = CBAM(inplanes=input_channels,planes=input_channels)
        self.cbam2 = nn.Sequential(CBAM(inplanes=input_channels,planes=input_channels),
                                    nn.Conv2d(input_channels,input_channels // 2,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(input_channels // 2),
                                    nn.ReLU())
        self.cbam3 = nn.Sequential(CBAM(inplanes=input_channels // 2,planes=input_channels //2),
                                    nn.Conv2d(input_channels // 2,input_channels // 4,kernel_size=(3,3),padding=1),
                                    nn.BatchNorm2d(input_channels // 4),
                                    nn.ReLU())

        self.transconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels,
                                            out_channels=input_channels // 2,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                        nn.BatchNorm2d(input_channels // 2),
                                        nn.ReLU())

        self.transconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels // 2,
                                            out_channels=input_channels // 4,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                        nn.BatchNorm2d(input_channels // 4),
                                        nn.ReLU())

        self.transconv3 = nn.Sequential(nn.ConvTranspose2d(in_channels = input_channels // 4,
                                            out_channels=input_channels // 8,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                            nn.BatchNorm2d(input_channels // 8),
                                            nn.ReLU())

        self.sd1 = Space2Depth(input_channels // 2, input_channels // 4)
        self.sd2 = Space2Depth(input_channels, input_channels // 2)
        self.sd3 = Space2Depth(input_channels // 2, input_channels // 2)
        
        

    def forward(self, encode_dict):
        e1 = encode_dict['e3'] # [2, 128, 4, 256]
        e2 = encode_dict['e2'] # [2, 64, 8, 512]
        e3 = encode_dict['e1'] # [2, 32, 16, 1024]

        batch_size = e1.shape[0]

        e1 = self.cbam1(e1) # [2, 128, 4, 256]
        e1 = self.transconv1(e1) # [2, 64, 8, 512]

        e2 = torch.cat([e1, e2], dim = 1) # [2, 128, 8, 512]
        e2 = self.cbam2(e2) # [2, 64, 8, 512]
        e2 = self.transconv2(e2) # [2, 32, 16, 1024]

        e3 = torch.cat([e2, e3], dim = 1)
        e3 = self.cbam3(e3)
        e3 = self.transconv3(e3) # [2, 16, 32, 2048]

        # Space2Depth + Conv 1x1
        sd1 = self.sd1(e3, down_scale = 2) # [2, 32, 304, 304]
        sd2 = self.sd2(sd1, down_scale = 2) # [2, 64, 152, 152]
        sd3 = self.sd3(sd2, down_scale = 1) # [2, 64, 152, 152]

        return sd3


class AL_3D(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, grid_size, muLti_task = True, **kwargs) -> None:
        super().__init__()

        self.model_version = model_cfg.MODEL_VERSION
        self.multi_task = model_cfg.MULTI_TASK
        self.pc_range = model_cfg.POINT_CLOUD_RANGE
        num_bev_seg_features = model_cfg.get('NUM_BEV_SEG_FEATURES', None)
        num_range_seg_features = model_cfg.get('NUM_RANGE_SEG_FEATURES', None)
        assert self.model_version in ['pillarNet_range', 'pillarNet_pw']
        if self.model_version == 'pillarNet_range':
            
            self.proj = Projection(model_cfg.POINT_CLOUD_RANGE, model_cfg.PC_FOV, model_cfg.BEV_SHAPE, model_cfg.RANGE_SHAPE)
            self.range_embed = nn.Linear(in_features=5, out_features=model_cfg.NUM_RANGE_FEATURES, bias=False)
            self.range_AL_2D = CP_Unet(input_channels=model_cfg.NUM_RANGE_FEATURES,output_channels=num_range_seg_features,layers_num=4,unet=self.multi_task)
            self.fusion = FusionBlock(input_channels=model_cfg.NUM_FUSION_FEATURES)

            if self.multi_task:
                self.classifier = nn.Sequential(
                    # self.act,
                    nn.Linear(model_cfg.NUM_BEV_SEG_FEATURES + model_cfg.NUM_RANGE_SEG_FEATURES,128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128,64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64,model_cfg.SEM_CLS))

        self.bev_AL_2D = CP_Unet(input_channels=model_cfg.NUM_BEV_FEATURES,output_channels=num_bev_seg_features,layers_num=4,unet=self.multi_task)
        

        if self.model_version == 'pillarNet_pw':

            sample_cfg = model_cfg.SAMPLE_PW_FEATURE
            self.sample_model = MLTSSD_Backbone(sample_cfg, num_class=num_class, input_channels=4)
            self.fusion = FusionBlockV2(input_channels=model_cfg.NUM_FUSION_FEATURES)

        self.num_point_features = 128

    def forward(self, batch_dict): 
        batch_dict['point_cloud_range'] = self.pc_range
        batch_size = batch_dict['batch_size']
        # 2D learning
        ori_bev = batch_dict['spatial_features']
        encode_bev, encode_bev_dict = self.bev_AL_2D(ori_bev)
        # prepare operations
        if hasattr(self,'sample_model'):
            batch_dict = self.sample_model(batch_dict)
            # viz.image(batch_dict['sample_pw_feature']['e3'][0,0,...].clamp(0,1))
            # feature fusion
            rv_fusion = self.fusion(batch_dict['sample_pw_feature'])

        if self.model_version == 'pillarNet_range':
            coord = batch_dict['points'][:,:4]
            points = batch_dict['points'][:,1:]
            range_pw_features = self.range_embed(points)

            keep_range = self.proj.init_range_coord(coord)[1]
            ori_range = self.proj.p2g_range(range_pw_features[keep_range], batch_size)
            # 2D learning
            encode_range, encode_range_dict = self.range_AL_2D(ori_range)
            encode_range_dict['coord'] = coord
            # feature fusion
            rv_fusion = self.fusion(encode_range_dict, self.proj)

        # creat det features
        det_features = torch.cat([encode_bev_dict['d0'], rv_fusion], dim = 1)
        batch_dict['spatial_features'] = det_features

        # viz.image(ori_bev[0,0,...].clamp(0,1), opts={'title': 'ori_bev1'})
        # viz.image(ori_bev[0,1,...].clamp(0,1), opts={'title': 'ori_bev2'})

        # viz.image(encode_bev_dict['e1'][0,0,...].clamp(0,1), opts={'title': 'encode_bev_e1'})
        # viz.image(encode_bev_dict['e2'][0,0,...].clamp(0,1), opts={'title': 'encode_bev_e2'})
        # viz.image(encode_bev_dict['e3'][0,0,...].clamp(0,1), opts={'title': 'encode_bev_e3'})

        # viz.image(encode_bev[0,0,...].clamp(0,1), opts={'title': 'encode_bev1'})
        # viz.image(encode_bev[0,1,...].clamp(0,1), opts={'title': 'encode_bev2'})

        # if hasattr(self,'sample_model'):
        #     viz.image(batch_dict['sample_pw_feature']['e1'][0,0,...].clamp(0,1), opts={'title': 'sample_pw_feature'})
        #     viz.image(batch_dict['sample_pw_feature']['e2'][0,0,...].clamp(0,1), opts={'title': 'sample_pw_feature'})
        #     viz.image(batch_dict['sample_pw_feature']['e3'][0,0,...].clamp(0,1), opts={'title': 'sample_pw_feature'})

        # viz.image(rv_fusion[0,1,...].clamp(0,1), opts={'title': 'rv_fusion'})
        

            
        if self.multi_task:
            # creat seg features
            # project & inv project
            keep_bev = self.proj.init_bev_coord(coord)[1]
            bev_pw = self.proj.g2p_bev(encode_bev)
            range_pw = self.proj.g2p_range(encode_range)

            c_bev = encode_bev.shape[1]
            c_range = encode_range.shape[1]

            cmplt_bev_pw = encode_bev.new_zeros([coord.shape[0], c_bev])
            cmplt_bev_pw[keep_bev] = bev_pw 

            cmplt_range_pw = encode_range.new_zeros([coord.shape[0], c_range])
            cmplt_range_pw[keep_range] = range_pw

            sem_pw_features = torch.cat([cmplt_bev_pw, cmplt_range_pw], dim = -1)
            sem_pred = self.classifier(sem_pw_features)
            batch_dict['sem_pred'] = sem_pred

        return batch_dict
        
