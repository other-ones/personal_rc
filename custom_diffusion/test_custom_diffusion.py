from collections import OrderedDict
import argparse
import sys
sys.path.insert(0,'./packages')
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers.loaders import AttnProcsLayers
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume_path')
    parser.add_argument('--modifier_token')
    parser.add_argument('--pretrained_model_name_or_path')
    args=parser.parse_args()
weight_dtype = torch.float32
pipe = DiffusionPipeline.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=weight_dtype
).to("cuda")

# Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
train_kv = True
args.freeze_model='crossattn_kv'
attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else CustomDiffusionAttnProcessor
    )
train_q_out = False if args.freeze_model == "crossattn_kv" else True
custom_diffusion_attn_procs = {}
st = pipe.unet.state_dict()
for name, _ in pipe.unet.attn_processors.items():
    cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = pipe.unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = pipe.unet.config.block_out_channels[block_id]
    layer_name = name.split(".processor")[0]
    weights = {
        "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
        "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
    }
    if train_q_out:
        weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
        weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
        weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
    if cross_attention_dim is not None:
        custom_diffusion_attn_procs[name] = attention_class(
            train_kv=train_kv,
            train_q_out=train_q_out,
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
        ).to(pipe.unet.device)
        custom_diffusion_attn_procs[name].load_state_dict(weights)
    else:
        custom_diffusion_attn_procs[name] = attention_class(
            train_kv=False,
            train_q_out=False,
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
        )
del st
pipe.unet.set_attn_processor(custom_diffusion_attn_procs)
custom_diffusion_layers = AttnProcsLayers(pipe.unet.attn_processors)
defined_state_dict=pipe.unet.state_dict()
saved_state_dict = torch.load(args.resume_path+'/custom_diffusion.pt', map_location=torch.device('cpu'))
new_state_dict={}
for key in defined_state_dict:
    if key in saved_state_dict:
        new_state_dict[key]=saved_state_dict[key]    
    else:
        new_state_dict[key]=defined_state_dict[key]    
pipe.unet.load_state_dict(new_state_dict,strict=True)
pipe.load_textual_inversion(args.resume_path,token="<new1>", weight_name="<new1>.bin")
image = pipe(
    "<new1> cat sitting in a bucket",
    num_inference_steps=100,
    guidance_scale=6.0,
    eta=1.0,
).images[0]
image.save("cat.png")