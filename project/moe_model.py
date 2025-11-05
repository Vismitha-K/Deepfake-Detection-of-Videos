import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MoEModel(nn.Module):
    def __init__(self, num_classes=2,
                 resnet_ckpt='checkpoints/resnet50/resnet50_best.pth',
                 mobilenet_ckpt='checkpoints/mobilenet_v3_large/mobilenet_v3_large_best.pth',
                 device='cpu'):
        super().__init__()
        self.device = device

        # -------------------- Expert 1: ResNet-50 --------------------
        resnet = models.resnet50(weights=None)  # newer torchvision syntax
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        try:
            ckpt_r = torch.load(resnet_ckpt, map_location='cpu')
            # If checkpoint is a dict with 'model_state' key, unwrap it
            if isinstance(ckpt_r, dict):
                if 'model_state' in ckpt_r:
                    ckpt_r = ckpt_r['model_state']
                elif 'state_dict' in ckpt_r:
                    ckpt_r = ckpt_r['state_dict']
            resnet.load_state_dict(ckpt_r, strict=False)
            print(f"[Info] Loaded ResNet checkpoint from {resnet_ckpt}")
        except Exception as e:
            print(f"[Warning] Could not load ResNet ckpt '{resnet_ckpt}': {e}")

        self.resnet = resnet.to(self.device)

        # -------------------- Expert 2: MobileNetV3-Large --------------------
        mobilenet = models.mobilenet_v3_large(weights=None)
        # MobileNet classifier: [Linear(960->1280), Hardswish, Dropout, Linear(1280->1000)]
        # We replace the final linear layer
        mobilenet.classifier[-1] = nn.Linear(
            mobilenet.classifier[-1].in_features, num_classes
        )

        try:
            ckpt_m = torch.load(mobilenet_ckpt, map_location='cpu')
            if isinstance(ckpt_m, dict):
                if 'model_state' in ckpt_m:
                    ckpt_m = ckpt_m['model_state']
                elif 'state_dict' in ckpt_m:
                    ckpt_m = ckpt_m['state_dict']
            mobilenet.load_state_dict(ckpt_m, strict=False)
            print(f"[Info] Loaded MobileNet checkpoint from {mobilenet_ckpt}")
        except Exception as e:
            print(f"[Warning] Could not load MobileNet ckpt '{mobilenet_ckpt}': {e}")

        self.mobilenet = mobilenet.to(self.device)

        # -------------------- Gating Network --------------------
        # ResNet pooled feature dim = 2048, MobileNet pooled feature dim = 960
        gate_input_dim = resnet.fc.in_features + 960

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        ).to(self.device)

        # Initialize bias so softmax ≈ [0.6, 0.4]
        with torch.no_grad():
            bias = torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32))
            self.gate[-1].bias.copy_(bias)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x: Tensor [B, 3, 224, 224]
        Returns dictionary with:
          'logits'       : final weighted logits [B, 2]
          'prob'         : softmax probabilities [B, 2]
          'weights'      : gate weights [B, 2]
          'expert_logits': tuple (resnet_logits, mobilenet_logits)
          'feat_maps'    : tuple (resnet_feature_map, mobilenet_feature_map)
        """

        # ----- ResNet forward -----
        r = self.resnet.conv1(x)
        r = self.resnet.bn1(r)
        r = self.resnet.relu(r)
        r = self.resnet.maxpool(r)
        r = self.resnet.layer1(r)
        r = self.resnet.layer2(r)
        r = self.resnet.layer3(r)
        r = self.resnet.layer4(r)
        r_feat_map = r
        r_pooled = torch.flatten(self.resnet.avgpool(r_feat_map), 1)
        r_logits = self.resnet.fc(r_pooled)

        # ----- MobileNet forward -----
        m_feat_map = self.mobilenet.features(x)
        m_pooled = torch.flatten(self.mobilenet.avgpool(m_feat_map), 1)
        m_logits = self.mobilenet.classifier(m_pooled)

        # ----- Gating network -----
        gate_in = torch.cat([r_pooled, m_pooled], dim=1)
        gate_logits = self.gate(gate_in)
        weights = F.softmax(gate_logits, dim=1)  # [B, 2]
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)

        # ----- Weighted combination -----
        final_logits = w1 * r_logits + w2 * m_logits

        return {
            'logits': final_logits,
            'prob': F.softmax(final_logits, dim=1),
            'weights': weights,
            'expert_logits': (r_logits, m_logits),
            'feat_maps': (r_feat_map, m_feat_map)
        }