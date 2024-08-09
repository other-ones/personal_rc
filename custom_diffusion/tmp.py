import torch
path='saved_models/custom_diffusion/single/dog6/custom_mlm01_dog6_train_te/checkpoints/checkpoint-250/<dog6>.bin'
out=torch.load(path)
print(out['<dog6>'].shape)
