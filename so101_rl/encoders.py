"""Image encoders for the SO-101 policies (Stable-Baselines3 features extractors).

``ResNetExtractor``: one ImageNet-pretrained ResNet18 per camera key. Each stacked history frame
goes through the camera's ResNet separately (shared weights); the final 512-channel feature map
keeps its spatial layout (where the cube is matters), is reduced by a 1x1 conv and projected to
``out_dim`` features per camera. The state vector is passed through unchanged.

With ``freeze=True`` (default) the ResNets keep their ImageNet weights and BatchNorm statistics;
only the projection heads and the policy train, which resists overfitting on 45 demonstrations.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        out_dim: int = 256,
        proj_channels: int = 64,
        freeze: bool = True,
    ):
        cameras = [k for k, s in observation_space.spaces.items() if len(s.shape) == 3]
        state_dim = sum(s.shape[0] for k, s in observation_space.spaces.items() if len(s.shape) == 1)
        super().__init__(observation_space, features_dim=len(cameras) * out_dim + state_dim)
        self.cameras = cameras
        self.state_keys = [k for k, s in observation_space.spaces.items() if len(s.shape) == 1]
        self.freeze = freeze
        self.backbones = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for key in cameras:
            channels, height, width = observation_space[key].shape  # channels-first after SB3 transpose
            if channels % 3:
                raise ValueError(f"{key}: expected stacked RGB frames, got {channels} channels")
            frames = channels // 3
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            body = nn.Sequential(*list(net.children())[:-2])  # drop avgpool + classifier
            if freeze:
                body.requires_grad_(False)
            with torch.no_grad():
                _, c, h, w = body.eval()(torch.zeros(1, 3, height, width)).shape
            self.backbones[key] = body
            self.heads[key] = nn.ModuleDict({
                "reduce": nn.Sequential(nn.Conv2d(c, proj_channels, 1), nn.ReLU()),
                "project": nn.Sequential(nn.Linear(frames * proj_channels * h * w, out_dim), nn.ReLU()),
            })
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def apply(self, fn):
        """Apply ``fn`` to the new layers only, never the pretrained backbones.

        SB3's ActorCriticPolicy runs ``features_extractor.apply(init_weights)`` (orthogonal init)
        after building the extractor, which would overwrite the ImageNet weights with random ones.
        """
        self.heads.apply(fn)
        fn(self)
        return self

    def matches_imagenet(self) -> bool:
        """True if every backbone still holds the ImageNet weights."""
        reference = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2])
        ref = reference.state_dict()
        return all(
            torch.equal(value.detach().cpu(), ref[name])
            for body in self.backbones.values()
            for name, value in body.state_dict().items()
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbones.eval()  # keep ImageNet BatchNorm statistics
        return self

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for key in self.cameras:
            x = observations[key]                                 # B, 3*frames, H, W in [0, 1]
            b, c, height, width = x.shape
            x = x.reshape(b * (c // 3), 3, height, width)
            x = (x - self.mean) / self.std
            # Frozen backbone: inference only, so run it in fp16 on the GPU's tensor cores.
            half = self.freeze and x.is_cuda
            with torch.set_grad_enabled(not self.freeze and torch.is_grad_enabled()), \
                    torch.autocast("cuda", dtype=torch.float16, enabled=half):
                fmap = self.backbones[key](x).float()             # B*frames, 512, h, w
            z = self.heads[key]["reduce"](fmap)                   # B*frames, proj, h, w
            parts.append(self.heads[key]["project"](z.reshape(b, -1)))
        for key in self.state_keys:
            parts.append(observations[key].float())
        return torch.cat(parts, dim=1)
