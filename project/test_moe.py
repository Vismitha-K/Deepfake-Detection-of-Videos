# test_moe.py
import torch
from moe_model import MoEModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
model = MoEModel(
    resnet_ckpt='checkpoints/resnet50/resnet50_best.pth',
    mobilenet_ckpt='checkpoints/mobilenet_v3_large/mobilenet_v3_large_best.pth',
    device='cpu'
)
model.eval()

x = torch.randn(2, 3, 224, 224).to(device)
out = model(x)
print("final logits shape:", out['logits'].shape)
print("gate weights:", out['weights'].detach().cpu().numpy())
print("example probs:", out['prob'].detach().cpu().numpy())