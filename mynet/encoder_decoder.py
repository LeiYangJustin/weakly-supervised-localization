import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from mynet.softargmax import SoftArgmax2D
import numpy as np

def compute_entropy(scores):
    ## logits: [B, K]
    logits = F.log_softmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    entropy = (probs * logits).sum(dim=-1).neg()
    return entropy

class Encoder(nn.Module):
    def __init__(self, resnet_type='resnet18'):
        super(Encoder, self).__init__()    
        if resnet_type == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.out_dim = 512
        elif resnet_type == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            self.out_dim = 512
        elif resnet_type == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.out_dim = 2048
        else:
            raise NotImplementedError
        
        self.pre_encode = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encode_layers = nn.ModuleList([
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])

    def forward(self, x):
        h = self.pre_encode(x)
        feature_list = []
        for idx, encode_layer in enumerate(self.encode_layers):
            h = encode_layer(h)
            feature_list.append(h)
            # print(idx, "layer: ", h.shape)
        return feature_list


class Decoder(nn.Module):
    def __init__(self, up_chl_list, use_residual_link=True):
        super(Decoder, self).__init__()
        
        self.use_residual_link = use_residual_link

        self.conv_up_list = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.dot_fc_list = nn.ModuleList()
        self.add_fc_list = nn.ModuleList()

        idx0 = up_chl_list[0]
        for idx1 in up_chl_list[1:]:
            ## idx0 -> idx1: channel size change
            ## stride = 2: upsample by 2
            self.conv_up_list.append(nn.ConvTranspose2d(idx0, idx1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False))
            self.dot_fc_list.append(nn.Conv2d(512, idx0, kernel_size=1, bias=False))
            self.add_fc_list.append(nn.Conv2d(512, idx0, kernel_size=1, bias=False))
            idx0 = idx1

        
    """
    h0: final feature map
    h1: second last feature map
    x: image
    """
    def forward(self, feature_list, memory):
        ## upsampling
        for idx, conv_up in enumerate(self.conv_up_list):
            if idx != 0:
                # print(idx, 'layer: conv_up', h.shape)
                # print(idx, 'layer: feat', feature_list[idx].shape)
                
                if self.use_residual_link:
                    h = h + feature_list[idx]
                
                if memory is not None:
                    dot_fc, add_fc = self.dot_fc_list[idx], self.add_fc_list[idx]
                    h = self.film_modulator(h, memory, dot_fc, add_fc)
            else:
                h = feature_list[idx]
            
            ## apply upscaling conv
            h = conv_up(h)

            ## relu applies to all features except the last one
            if idx < len(self.conv_up_list)-1:
                h = self.relu(h)

        return h

    """
    https://distill.pub/2018/feature-wise-transformations/
    feat: src feature map to be modulated
    memory: memory for modulation
    """
    def film_modulator(self, feat, memory, dot_fc, add_fc):
        B, C = memory.shape
        memory = memory.reshape(B, C, 1, 1)
        return dot_fc(memory)*feat+add_fc(memory)


"""
Spatial Broadcast Decoder Module
https://arxiv.org/pdf/1901.07017.pdf 
"""
class SBDecoder(nn.Module):
    def __init__(self, z_dim, img_dim, out_channels, pos_dim=0):
        super(SBDecoder, self).__init__()

        self.H, self.W = img_dim
        x_range = torch.linspace(-1.,1.,self.W)
        y_range = torch.linspace(-1.,1.,self.H)
        x_grid, y_grid = torch.meshgrid([x_range,y_range])
        x_grid = x_grid.expand(1,1,-1,-1).clone()
        y_grid = y_grid.expand(1,1,-1,-1).clone()

        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(z_dim+2,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ELU(),
            torch.nn.Conv2d(64,out_channels,kernel_size=3,stride=1,padding=1))

        # ## Additional MLP Layer -> according to KG should improve reconstruction quality at cost of disentanglement
        # if pos_dim != 0:
        #     self.pre_mlp = torch.nn.Sequential(
        #         torch.nn.Linear(512,256),
        #         torch.nn.ELU(),
        #         torch.nn.Linear(256,pos_dim),
        #         torch.nn.ELU())

    def forward(self, feature_list, memory):
        z = memory
        N = z.shape[0]
        # z = self.pre_mlp(z)

        z_broad = z.view(z.shape + (1,1))
        z_broad = z_broad.expand(-1,-1,self.H,self.W)

        vol = torch.cat(
            (self.x_grid.expand(N,-1,-1,-1),
            self.y_grid.expand(N,-1,-1,-1),
            z_broad), dim=1)
        conv_out = self.conv_layer(vol)

        mu_x = torch.sigmoid(conv_out[:,:3,:,:])
        ret = torch.cat((mu_x,conv_out[:,(3,),:,:]),dim=1)
        return ret


class EDModel(nn.Module):
    def __init__(self, output_dim, resnet_type='resnet18', up_chl_list = [512, 256, 128, 64, 4]):
        super(EDModel, self).__init__()
        self.output_dim = output_dim ## NUM OF CLASSES

        self.decoder = Decoder(up_chl_list=up_chl_list, use_residual_link=True) ## proposal net
        # self.decoder = SBDecoder(512, (64, 64), 4)

        kernel_size, padding = 1, 0
        self.rgb_fc = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=padding) ## or kernel_size=5 (no good), 1 (good)
        self.alpha_fc = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding)
        
        self.embeddings = nn.Embedding(self.output_dim, 512) ## object embeddings
        self.encoder = Encoder(resnet_type=resnet_type) ## conv encoder

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) ## average pooling
        self.dropout = nn.Dropout(p=0.2)

        self.b = torch.ones((1, self.output_dim))
        self.max_ent = compute_entropy(self.b)


    def compute_entropy_weight(self, scores):
        entropy = compute_entropy(scores)
        entropy_weight = 1.0 - entropy/self.max_ent.to(scores.device)
        return entropy_weight

    """
    INPUT:
        img: [B, 3, 224, 224]
    OUTPUT:
        h: [B, K]
        attns: [B, 7, 7, K]
        object_masks: [B, 3, 112, 112]
        feat: [B, C, 7, 7]
    """
    def forward(self, img, gt_label):
        feature_list = self.encoder(img)
        B, C, H, W = feature_list[-1].shape
        feat = feature_list[-1]

        ## activations
        x = feat.reshape(B, C, -1) ## [B, C, H, W] -> [B, C, HW]
        x = self.dropout(x)
        x = torch.matmul(x.transpose(1,2), self.embeddings.weight.t())   ## [B, HW, K], activation map
        attns = F.softmax(x, dim=1).permute(2,0,1).reshape(self.output_dim, B, H, W) ## [K, B, H, W]

        ## classification
        x = x.permute(0,2,1).reshape(B, self.output_dim, H, W)  ## output feature
        h = self.avg_pool(x).reshape(B, self.output_dim)        ## class logits: [B, K]
        
        if gt_label is None:
            return h, attns
            
        ## reconstruction
        reversed_list = feature_list[::-1] ## reversed list
        memory = self.embeddings(gt_label)
        object_output = self.decoder(reversed_list, memory=memory) ## [B, 3+1, H, W]; RGB+ALPHA
        object_alphas = F.sigmoid(self.alpha_fc(object_output[:, -1:]))
        object_masks = F.sigmoid(self.rgb_fc(object_output[:, :-1]))

        return h, attns, object_masks, object_alphas
        # else:
        #     bg_memory = self.embeddings(torch.zeros_like(gt_label)+self.output_dim-1)
        #     bg_masks = self.decoder(reversed_list, bg_memory) ## [B, 3+1, H, W]; RGB+ALPHA

        #     bg_alphas = bg_masks[:,-1:]
        #     bg_masks = bg_masks[:,:-1]
        #     bg_masks = F.sigmoid(bg_masks)

        #     alpha = torch.cat([object_alphas, bg_alphas], dim=1)
        #     alpha = F.softmax(alpha, dim=1)
        #     return h, attns, object_masks, bg_masks, alpha
