import sys
sys.path.insert(0, './packages')
# from diffusers.models.vae import Encoder
import torch 
from torch import nn
import pdb
class FeedForward(nn.Module):
    def __init__(self, embed_dim, width_fac=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.W_ff1 = nn.Linear(embed_dim, width_fac * embed_dim)
        self.W_ff2 = nn.Linear(embed_dim * width_fac, embed_dim)
        self.relu = nn.ReLU()
    def forward(self, X):
        # Simple Feedforward network that projects into a higher space (by width_fac) and back to embed_dim
        X = self.W_ff1(X)
        X = self.relu(X)
        return self.W_ff2(X)
class ContextNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        max_position_embeddings=77,
    ):
        super().__init__()
    
        # self.net = Encoder(
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
        #     block_out_channels=(32,64,64,128),
        #     layers_per_block=2,
        #     norm_num_groups=16,
        #     act_fn="silu",
        #     double_z=False,
        # )
        
        self.final = nn.Linear(in_channels, out_channels)
        # self.read_tkns = nn.Parameter(torch.empty(1, max_char, 128))
        # self.read_tkns.data.normal_(0.0, 0.02)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.position_embedding = nn.Embedding(77, in_channels)
        # self.ca_layers = nn.ModuleList()



        self.multi_head_attention1 = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        self.feed_forward1 = FeedForward(embed_dim=in_channels)
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.multi_head_attention2 = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        # for _ in range(4):
        #     self.ca_layers.append(nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True))


    def forward(self, x,position_ids=None):
        # x = self.net(x).flatten(2).permute(0,2,1) 
        # read_tkns = self.read_tkns.repeat(x.shape[0], 1, 1)
        # x=x.permute(0,2,1)
        seq_length = x.shape[-1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
    
        position_embeddings = self.position_embedding(position_ids)
        x=x+position_embeddings
        # for layer in self.ca_layers:
        #     x, _ = layer(x, x, x)
        out,_=self.multi_head_attention1(x, x, x)
        x+=out
        x = self.layer_norm1(x) # add residual connection + layer_norm
        x=self.feed_forward1(x)
        out,_=self.multi_head_attention2(x, x, x)
        x+=out
        x = self.layer_norm2(x) # add residual connection + layer_norm
        # torch.Size([150, 77, 768]) x.shape1
        x = self.final(x)
        # torch.Size([150, 77, 49410]) x.shape2
        return x

if __name__ == '__main__':
    net = ContextNet(768, 40393).cuda()
    data = torch.randn(10, 77, 768).cuda()
    print(net(data).shape)