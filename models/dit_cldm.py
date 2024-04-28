import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from models.dit import DiTBlock, TimestepEmbedder, LabelEmbedder, FinalLayer, get_2d_sincos_pos_embed


class DiTControlNet(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.control_depth = depth // 2
        self.controlnet = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(self.control_depth)
        ])
        self.token_num = (input_size // patch_size) ** 2
        self.zero_convs = nn.ModuleList([
            self.make_zero_conv(self.token_num) for _ in range(self.control_depth)
        ])

    def make_zero_conv(self, channels):
        conv = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return nn.Sequential(conv)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward_c(self, c):
        self.h, self.w = c.shape[-2] // self.patch_size, c.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.h)
        ).unsqueeze(0).to(c.device).to(self.dtype)
        return self.x_embedder(c) + pos_embed if c is not None else c

    def forward(self, x, t, y, c, **kwargs):
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c)

        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        emb = t + y

        outs = []

        h = c.type(self.dtype)
        h += x
        for module, zero_conv in zip(self.controlnet, self.zero_convs):
            h = module(h, emb)
            outs.append(zero_conv(h))

        with torch.no_grad():
            for i, module in enumerate(self.blocks):
                if i < self.control_depth:
                    x = module(x, emb)
                    x += outs.pop(0)
                else:
                    x = module(x, emb)

        x = self.final_layer(x, emb)
        x = self.unpatchify(x)
        return x


    @property
    def dtype(self):
        return next(self.parameters()).dtype
