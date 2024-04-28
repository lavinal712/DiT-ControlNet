import torch

from models.dit_cldm import DiTControlNet


DIT_MODEL_PATH = "E:/DiT/DiT-XL-2-256/DiT-XL-2-256x256.pt"
DITCONTROLNET_MODEL_PATH = "E:/DiT-ControlNet/DiTControlNet-XL-2-256x256.pt"

pretrained_weights = torch.load(DIT_MODEL_PATH)

model = DiTControlNet()
scratch_dict = model.state_dict()
target_dict = {}

for k in scratch_dict.keys():
    if k in pretrained_weights:
        target_dict[k] = pretrained_weights[k].clone()
    elif 'controlnet' in k:
        copy_k = k.replace('controlnet', 'blocks')
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), DITCONTROLNET_MODEL_PATH)
print('Done.')
