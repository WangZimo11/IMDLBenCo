import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class Multiple(nn.Module):
    def __init__(self, 
                 init_value = 1e-6,
                 embed_dim = 512,
                 predict_channels = 1,
                 norm_layer = partial(nn.LayerNorm, eps=1e-6) ):
        super(Multiple, self).__init__()
        self.gamma1 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma2 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma3 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma4 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma5 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        self.gamma6 = nn.Parameter(init_value * torch.ones((embed_dim)),requires_grad=True)
        # self.drop_path = nn.Identity()
        self.norm = norm_layer(embed_dim)
        
        self.conv_layer1 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer3 = nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_layer4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)
    def forward(self, x):
        c1, c2, c3, c4 = x
        
        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)
        
        b, c, h, w = c1.shape
        c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False)
        
        
        c1 = c1.flatten(2).transpose(1, 2)
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2) 
        x = self.gamma1*c1 + self.gamma2*c2 + self.gamma3*c3 + self.gamma4*c4
        x= x.transpose(1, 2).reshape(b, c, h, w)
        x = (self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()
        x = self.conv_last(x)
        return x
    
    
    
    
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
    
class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        n1_in_channels, n2_in_channels, n3_in_channels, n4_in_channels = [32, 64, 160, 256]

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        # self.linear_n4 = MLP(input_dim=n4_in_channels, embed_dim=embedding_dim)
        # self.linear_n3 = MLP(input_dim=n3_in_channels, embed_dim=embedding_dim)
        # self.linear_n2 = MLP(input_dim=n2_in_channels, embed_dim=embedding_dim)
        # self.linear_n1 = MLP(input_dim=n1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    # def forward(self, image_inputs, noise_inputs):
    def forward(self, image_inputs):
        c1, c2, c3, c4 = image_inputs
        # n1, n2, n3, n4 = noise_inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        # _n4 = self.linear_n4(n4).permute(0,2,1).reshape(n, -1, n4.shape[2], n4.shape[3])
        # _n4 = F.interpolate(_n4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # _n3 = self.linear_n3(n3).permute(0,2,1).reshape(n, -1, n3.shape[2], n3.shape[3])
        # _n3 = F.interpolate(_n3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        # _n2 = self.linear_n2(n2).permute(0,2,1).reshape(n, -1, n2.shape[2], n2.shape[3])
        # _n2 = F.interpolate(_n2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _n1 = self.linear_n1(n1).permute(0,2,1).reshape(n, -1, n1.shape[2], n1.shape[3])
        # _n1 = F.interpolate(_n1, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1, _n4, _n3, _n2, _n1], dim=1))
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x