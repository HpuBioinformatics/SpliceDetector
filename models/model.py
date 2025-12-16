import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ECAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAttention, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float)) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class BiologicalAttention(nn.Module):
    def __init__(self, channels, seq_length=402, bio_weight=0.5):
        super(BiologicalAttention, self).__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.bio_weight = bio_weight
        self.motif_weights = nn.Parameter(torch.randn(1, 1, seq_length))
        self.distance_conv_near = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.distance_conv_mid = nn.Conv1d(channels, channels, 21, padding=10, groups=channels)
        self.distance_conv_far = nn.Conv1d(channels, channels, 51, padding=25, groups=channels)
        self.fusion_conv = nn.Conv1d(channels * 3, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, length = x.shape
        near_features = self.distance_conv_near(x)
        mid_features = self.distance_conv_mid(x)
        far_features = self.distance_conv_far(x)
        multi_scale = torch.cat([near_features, mid_features, far_features], dim=1)
        distance_weights = self.sigmoid(self.fusion_conv(multi_scale))
        if length <= self.seq_length:
            motif_weights = self.motif_weights[:, :, :length]
        else:
            motif_weights = F.interpolate(self.motif_weights, size=length, mode='linear', align_corners=False)
        bio_weights = distance_weights * (1 + self.bio_weight * motif_weights)
        
        return x * bio_weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention_type='eca', pool_size=1, 
                 bio_weight=0.5, reduction=16):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if attention_type == 'eca':
            self.attention = ECAttention(out_channels)
        elif attention_type == 'cbam':
            self.attention = CBAM(out_channels, reduction=reduction)
        else:
            self.attention = nn.Identity()
        
        self.bio_attention = BiologicalAttention(out_channels, bio_weight=bio_weight)
        
        self.pool = nn.AvgPool1d(pool_size) if pool_size > 1 else nn.Identity()
        
        if in_channels != out_channels or pool_size > 1:
            shortcut_layers = [nn.Conv1d(in_channels, out_channels, 1, bias=False)]
            if pool_size > 1:
                shortcut_layers.append(nn.AvgPool1d(pool_size))
            shortcut_layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        out = self.attention(out)
        out = self.bio_attention(out)
        
        out += identity
        return self.relu(out)

class LightweightTransformer(nn.Module):
    def __init__(self, dim, depth=2, heads=8, dim_head=32, mlp_dim=512, dropout=0.1):
        super(LightweightTransformer, self).__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        batch_size, channels, length = x.shape
        x = x.transpose(1, 2)
        if length <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :length, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2), 
                size=length, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_emb
        x = self.transformer(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x

class SpliceDetectorBase(nn.Module):
    def __init__(self, attention_type='eca', bio_weight=0.5, reduction=16, 
                 use_transformer=False, num_classes=2, dropout=0.1):
        super(SpliceDetectorBase, self).__init__()
        
        self.use_transformer = use_transformer
        self.initial_conv = nn.Conv1d(4, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ConvBlock(64, 64, attention_type=attention_type, pool_size=1, 
                               bio_weight=bio_weight, reduction=reduction)
        self.block2 = ConvBlock(64, 128, attention_type=attention_type, pool_size=3, 
                               bio_weight=bio_weight, reduction=reduction)
        self.block3 = ConvBlock(128, 128, attention_type=attention_type, pool_size=4, 
                               bio_weight=bio_weight, reduction=reduction)
        self.block4 = ConvBlock(128, 256, attention_type=attention_type, pool_size=4, 
                               bio_weight=bio_weight, reduction=reduction)
        
        if use_transformer:
            self.transformer = LightweightTransformer(256, depth=2, heads=8, dropout=dropout)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 3 and x.size(-1) == 4:
            x = x.transpose(1, 2)  
        
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        if self.use_transformer:
            x = self.transformer(x)
    
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        
        return x

class SpliceDetectorModelA(SpliceDetectorBase):
    def __init__(self, num_classes=2):
        super(SpliceDetectorModelA, self).__init__(
            attention_type='eca',
            bio_weight=0.5,
            reduction=16,
            use_transformer=False,
            num_classes=num_classes,
            dropout=0.3
        )

class SpliceDetectorModelB(SpliceDetectorBase):
    def __init__(self, num_classes=2):
        super(SpliceDetectorModelB, self).__init__(
            attention_type='eca',
            bio_weight=0.8,
            reduction=8,
            use_transformer=True,
            num_classes=num_classes,
            dropout=0.2
        )

class SpliceDetectorModelC(SpliceDetectorBase):
    def __init__(self, num_classes=2):
        super(SpliceDetectorModelC, self).__init__(
            attention_type='cbam',
            bio_weight=0.65,
            reduction=12,
            use_transformer=False,
            num_classes=num_classes,
            dropout=0.25
        )

class SpliceDetectorEnsemble(nn.Module):
    def __init__(self, num_classes=2, ensemble_weights=None):
        super(SpliceDetectorEnsemble, self).__init__()
        
        self.model_a = SpliceDetectorModelA(num_classes)
        self.model_b = SpliceDetectorModelB(num_classes)
        self.model_c = SpliceDetectorModelC(num_classes)
        if ensemble_weights is None:
            ensemble_weights = [0.4, 0.35, 0.25]  
        self.register_buffer('ensemble_weights', torch.tensor(ensemble_weights))
        
    def forward(self, x, return_individual=False):
        pred_a = self.model_a(x)
        pred_b = self.model_b(x)
        pred_c = self.model_c(x)
        
        if return_individual:
            return pred_a, pred_b, pred_c

        prob_a = F.softmax(pred_a, dim=1)
        prob_b = F.softmax(pred_b, dim=1)
        prob_c = F.softmax(pred_c, dim=1)
        
        ensemble_prob = (self.ensemble_weights[0] * prob_a + 
                        self.ensemble_weights[1] * prob_b + 
                        self.ensemble_weights[2] * prob_c)
        
        ensemble_logits = torch.log(ensemble_prob + 1e-8)
        
        return ensemble_logits
    
    def update_ensemble_weights(self, weights):
        self.ensemble_weights = torch.tensor(weights).to(self.ensemble_weights.device)
    
    def predict_with_confidence(self, x):
        pred_a, pred_b, pred_c = self.forward(x, return_individual=True)
        
        prob_a = F.softmax(pred_a, dim=1)
        prob_b = F.softmax(pred_b, dim=1)
        prob_c = F.softmax(pred_c, dim=1)
        
        conf_a = prob_a.max(dim=1)[0]
        conf_b = prob_b.max(dim=1)[0]
        conf_c = prob_c.max(dim=1)[0]
        
        total_conf = conf_a + conf_b + conf_c
        dynamic_weights = torch.stack([conf_a, conf_b, conf_c], dim=1) / total_conf.unsqueeze(1)
        
        weighted_prob = (dynamic_weights[:, 0:1] * prob_a + 
                        dynamic_weights[:, 1:2] * prob_b + 
                        dynamic_weights[:, 2:3] * prob_c)
        
        return weighted_prob, (conf_a, conf_b, conf_c)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_SpliceDetector_models(num_classes=2):
    model_a = SpliceDetectorModelA(num_classes)
    model_b = SpliceDetectorModelB(num_classes)  
    model_c =SpliceDetectorModelC(num_classes)
    
    return model_a, model_b, model_c

def create_SpliceDetector_ensemble(num_classes=2, ensemble_weights=None):
    return SpliceDetectorEnsemble(num_classes, ensemble_weights)

