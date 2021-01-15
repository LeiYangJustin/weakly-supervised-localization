import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models


def compute_entropy(scores):
    ## logits: [B, N]
    logits = F.log_softmax(scores, dim=-1)
    probs = F.softmax(scores, dim=-1)
    entropy = (probs * logits).sum(dim=-1).neg()
    return entropy


## ResNet18
class ResNetBase(nn.Module):
    def __init__(self, resnet_type='resnet18'):
        super(ResNetBase, self).__init__()    
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

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x):
        return self.features(x)


class MyNet(nn.Module):
    def __init__(self, output_dim):
        super(MyNet, self).__init__()
        self.output_dim = output_dim

        self.resnet = ResNetBase(resnet_type='resnet34')
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) ## average pooling
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(512, output_dim)

        self.b = torch.ones((1, output_dim))
        self.max_ent = compute_entropy(self.b)


    def forward(self, x):
        B = x.shape[0]
        h = self.resnet(x)
        h = self.avg_pool(h).reshape(B, 512)
        h = self.dropout(h)
        return self.head(h)


    def forward_eval(self, x):
        B = x.shape[0]
        x = self.resnet(x)
        h = self.avg_pool(x).reshape(B, 512)
        return self.head(h), x
        

    def detection(self, x):
        x = self.resnet(x)
        B,C,H,W = x.shape
        x = x.reshape(B, C, -1) ## [B, C, H, W] -> [B, C, HW]

        fc_weights = self.head.weight.data ## [N, C]
        activations = []
        for hc in fc_weights:
            hc = hc.reshape(1, C, 1)
            a = (hc * x).sum(dim=1)
            a = F.gumbel_softmax(a, dim=-1)
            # a = F.softmax(a, dim=-1)
            activations.append(a.reshape(B, H, W))
        return torch.stack(activations, dim=1)

    def compute_entropy_weight(self, scores):
        entropy = compute_entropy(scores)
        entropy_weight = 1.0 - entropy/self.max_ent.to(scores.device)
        return entropy_weight


class Decoder(nn.Module):
    def __init__(self, up_chl_list = [512, 256, 64], regression=4):
        super(Decoder, self).__init__()

        self.regression = regression
        self.conv_up_list = nn.ModuleList()
        
        idx0 = up_chl_list[0]
        for idx1 in up_chl_list[1:]:
            ## idx0 -> idx1: channel size change
            ## stride = 2: upsample by 2
            self.conv_up_list.append(nn.ConvTransposed2d(idx0, idx1, kernel_size=3, stride=2, padding=1, bias=False))
            idx0 = idx1

        if self.regression is not None:
            self.regression_head = nn.Conv1d(up_chl_list[-1]+3, regression, kernel_size=1)

        
    """
    h0: final feature map
    h1: second last feature map
    x: image
    """
    def forward(self, feature_list, image):
        feature_list = feature_list + image
        ## upsampling
        for idx, conv_up in enumerate(self.conv_up_list):
            if idx != 0:
                h = h + feature_list[idx]
            h = conv_up(feature_list[idx])

        if self.regression is None:
            return h
        else:
            return self.regression_head(h)

        

            
