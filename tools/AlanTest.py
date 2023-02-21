from pcdet.models.backbones_2d import convmlp
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=nn.BatchNorm2d, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out = out + identity
        out = self.relu(out)

        return out

class BasicBlock_CP(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, dilation, padding, stride = 1) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=input_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                padding=padding,
                                stride=stride)
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
        

class EncBlock(nn.Module):
    def __init__(self,input_channels) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = int(input_channels * 2)
        self.conv1 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(3, 3),
                                dilation=1,
                                padding=1,
                                stride=1)

        self.conv2 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(3, 3),
                                dilation=2,
                                padding=2,
                                stride=1)

        self.conv3 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(2,2),
                                dilation=2,
                                padding=1,
                                stride=1)

        self.conv4 = BasicBlock_CP(input_channels=self.input_channels*3,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

        self.conv5 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)
        self.pool = nn.AvgPool2d(kernel_size=(2,2),stride=2)

    def forward(self,x):
        input_data = x
        output_1 = self.conv1(input_data) # [4, 64, 512, 512]
        output_2 = self.conv2(output_1) # [4, 64, 512, 512]
        output_3 = self.conv3(output_2) # [4, 64, 512, 512]

        output_123 = torch.cat([output_1,output_2,output_3], dim = 1) # [4, 192, 512, 512]

        output_123_1 = self.conv4(output_123) # [4, 128, 512, 512]
        output_1_1 = self.conv5(x) # [4, 128, 512, 512]

        output = output_123_1 + output_1_1

        output = self.pool(output) # [4, 128, 256, 256]

        return output

class DecBlock(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // 2

        self.transconv = nn.Sequential(nn.ConvTranspose2d(in_channels = self.input_channels,
                                            out_channels=self.output_channels,
                                            kernel_size=(3,3),
                                            padding=1,
                                            stride=2,
                                            output_padding=1),
                                        nn.BatchNorm2d(self.output_channels),
                                        nn.ReLU())

        self.conv1 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(3,3),
                                dilation=1,
                                padding=1)

        self.conv2 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(3,3),
                                dilation=2,
                                padding=2)

        self.conv3 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(2,2),
                                dilation=2,
                                padding=1)

        self.conv4 = BasicBlock_CP(input_channels=self.output_channels*3,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

        self.conv5 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

    def forward(self, x):
        ouput_1 = self.transconv(x) # [4, 64, 511, 511]
        output_2 = self.conv1(ouput_1) # [4, 64, 511, 511]
        output_3 = self.conv2(output_2) # [4, 64, 511, 511]
        output_4 = self.conv3(output_3) # [4, 64, 511, 511]
        output_234 = torch.cat([output_2, output_3, output_4], dim = 1) # [4, 192, 511, 511]
        output_234 = self.conv4(output_234) # [4, 64, 511, 511]
        output_1_1 = self.conv5(ouput_1) # [4, 64, 511, 511]
        output = output_234 + output_1_1

        return output
        

class CP_Unet(nn.Module): # layers_num = 4 in our project
    def __init__(self, input_channels, layers_num, output_channels) -> None:
        super().__init__()
        self.layers = [int(input_channels * 2**i) for i in range(layers_num)]

        self.pre_conv = BasicBlock(inplanes=input_channels, planes=input_channels)
        self.out_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.encode_blocks = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        self.basic_blocks = nn.ModuleList()

        for i in range(len(self.layers) - 1):
            self.encode_blocks.append(EncBlock(input_channels=self.layers[i]))
            self.decode_blocks.append(DecBlock(input_channels=self.layers[0-1-i]))
            self.basic_blocks.append(BasicBlock(inplanes=self.layers[-1-i], planes=self.layers[-2-i]))
    
    def forward(self,x):
        e0 = self.pre_conv(x) # [4, 16, 512, 512]
        
        e1 = self.encode_blocks[0](e0) # [4, 32, 256, 256]
        e2 = self.encode_blocks[1](e1) # [4, 64, 128, 128]
        e3 = self.encode_blocks[2](e2) # [4, 128, 64, 64]

        d0 = self.decode_blocks[0](e3) # [4, 64, 128, 128]
        d0 = torch.cat([e2, d0], dim = 1) # [4, 128, 128, 128]
        d0 = self.basic_blocks[0](d0) # [4, 64, 128, 128]

        d1 = self.decode_blocks[1](d0) # [4, 32, 256, 256]
        d1 = torch.cat([e1, d1], dim = 1) # [4, 64, 256, 256]
        d1 = self.basic_blocks[1](d1) # [4, 32, 256, 256]

        d2 = self.decode_blocks[2](d1) # [4, 16, 512, 512]
        d2 = torch.cat([e0, d2], dim = 1) # [4, 32, 512, 512]
        d2 = self.basic_blocks[2](d2) # [4, 16, 512, 512]

        out = self.out_conv(d2)

        return out


class RB_Fusion(nn.Module):
    expansion = 1

    def __init__(self, input_channels=None):
        super(RB_Fusion, self).__init__()

        bev_feature_dim = 256
        range_feature_dim = 64

        self.channel_avg_func = nn.AdaptiveAvgPool2d((1,1))
        self.channel_max_func = nn.AdaptiveMaxPool2d((1,1))
        self.channel_ln = nn.Sequential(nn.Linear(in_features=(bev_feature_dim + range_feature_dim) * 2, 
                                                out_features=bev_feature_dim,
                                                bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(in_features=bev_feature_dim,
                                                out_features=bev_feature_dim + range_feature_dim),
                                        nn.ReLU())
                                                
        self.space_ln = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), stride=1, padding=1),
                                        nn.ReLU())
        

        self.act = nn.Sigmoid()

        self.num_bev_features = bev_feature_dim + range_feature_dim
        self.bev_feature_dim = bev_feature_dim
        self.range_feature_dim = range_feature_dim

    def forward(self, x):
        # x = batch_dict['spatial_features']
        
        bev_feature = x[:,:self.bev_feature_dim,...] # [4, 256, 152, 152]
        range_feature = x[:,self.bev_feature_dim:,...] # [4, 64, 152, 152]
         
        # bev
        bev_channel_avg = self.channel_avg_func(bev_feature).squeeze() # [4, 256]
        bev_channel_max = self.channel_max_func(bev_feature).squeeze() # [4, 256]

        bev_space_avg = torch.mean(bev_feature, dim = 1).unsqueeze(dim = 1) # [4, 1, 152, 152]
        bev_space_max = torch.max(bev_feature, dim = 1)[0].unsqueeze(dim = 1) # [4, 1, 152, 152]

        # range
        range_channel_avg = self.channel_avg_func(range_feature).squeeze() # [4, 64]
        range_channel_max = self.channel_max_func(range_feature).squeeze() # [4, 64]

        range_space_avg = torch.mean(range_feature, dim = 1).unsqueeze(dim = 1) # [4, 1, 152, 152]
        range_space_max = torch.max(range_feature, dim = 1)[0].unsqueeze(dim = 1) # [4, 1, 152, 152]

        # attention map
        channel_wise = torch.cat([bev_channel_avg, range_channel_avg, bev_channel_max, range_channel_max], dim = -1) # [4, 640]
        space_wise = torch.cat([bev_space_avg, range_space_avg, bev_space_max, range_space_max], dim = 1) # [4, 4, 152, 152]

        channel_wise = self.channel_ln(channel_wise).unsqueeze(dim = -1).unsqueeze(dim = -1) # [4, 320, 1, 1]
        space_wise = self.space_ln(space_wise).repeat(1,channel_wise.shape[1],1,1) # [4, 320, 152, 152]
        attention_map = channel_wise * space_wise # [4, 320, 152, 152]
        attention_map = self.act(attention_map)

        out = attention_map*x + x
        
        # batch_dict['spatial_features_2d'] = out

        return out

        

        



        





if __name__ == "__main__":
    data = torch.randn(4,320,152,152)
    # model = convmlp.convmlp_s()
    # model = EncBlock(input_channels=64)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of params: {n_parameters}')
    # output = model(data)
    # print(output.shape)

    # model = DecBlock(input_channels = output.shape[1])
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of params: {n_parameters}')
    # output = model(output)
    # print(output.shape)

    # model = CP_Unet(input_channels=64,layers_num=4,output_channels=64)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of params: {n_parameters}')
    # output = model(data)
    # print(output.shape)

    model = RB_Fusion()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of params: {n_parameters}')
    output = model(data)
    print(output.shape)
