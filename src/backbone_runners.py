"""
Unified backbone interface for SAE feature explorer inference.

Provides:
  - BackboneRunner / DINOv3Runner / CLIPRunner  for single-image on-the-fly inference
  - load_batched_backbone()  for batched precompute scripts
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms as trn

_DINO_MEAN = [0.485, 0.456, 0.406]
_DINO_STD  = [0.229, 0.224, 0.225]


class BackboneRunner:
    """Abstract backbone runner."""
    d_hidden: int
    device: torch.device

    def get_patch_tokens(self, img: Image.Image, image_size: int, token_type: str) -> torch.Tensor:
        """
        Run the backbone on a single PIL image.

        Returns a (n_tokens, d_hidden) float32 tensor on self.device.
        For spatial token_type: n_tokens = n_patches.
        For cls token_type: n_tokens = 1.
        """
        raise NotImplementedError


def _dino_transform(image_size: int) -> trn.Compose:
    return trn.Compose([
        trn.Resize((image_size, image_size),
                   interpolation=trn.InterpolationMode.BICUBIC,
                   antialias=True),
        trn.ToTensor(),
        trn.Normalize(_DINO_MEAN, _DINO_STD),
    ])


class DINOv2Runner(BackboneRunner):
    """DINOv2 ViT-B/14.  No register tokens; token layout: [CLS, patches]."""

    def __init__(self, device: torch.device, layer: int | None = None):
        self.device = device
        self._layer = layer
        print(f"Loading DINOv2 ViT-B/14 on {device}...")
        self._model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        ).to(device).eval()
        self.d_hidden = 768

        self._activation = [None]
        if layer is not None:
            def _hook(module, input, output):
                self._activation[0] = output
            self._model.blocks[layer].register_forward_hook(_hook)

    def get_patch_tokens(self, img: Image.Image, image_size: int, token_type: str) -> torch.Tensor:
        t = _dino_transform(image_size)(img).unsqueeze(0).to(self.device)
        self._activation[0] = None
        with torch.no_grad():
            out = self._model.forward_features(t)

        if self._layer is not None:
            hs = self._activation[0]  # (1, 1 + n_patches, 768)
        else:
            cls     = out['x_norm_clstoken'].unsqueeze(1)   # (1, 1, 768)
            patches = out['x_norm_patchtokens']              # (1, n_patches, 768)
            hs = torch.cat([cls, patches], dim=1)

        if token_type == 'cls':
            return hs[:, 0:1, :].reshape(-1, self.d_hidden)
        return hs[:, 1:, :].reshape(-1, self.d_hidden)


class DINOv3Runner(BackboneRunner):
    def __init__(self, device: torch.device):
        from transformers import AutoModel
        self.device = device
        print(f"Loading DINOv3 backbone on {device}...")
        self._model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            torch_dtype=torch.float32,
        ).to(device).eval()
        self.d_hidden = self._model.config.hidden_size
        self._n_reg   = self._model.config.num_register_tokens

    def get_patch_tokens(self, img: Image.Image, image_size: int, token_type: str) -> torch.Tensor:
        t   = _dino_transform(image_size)(img).unsqueeze(0).to(self.device)
        out = self._model(pixel_values=t)
        hs  = out.last_hidden_state  # (1, 1 + n_reg + n_patches, d)
        if token_type == 'cls':
            return hs[:, 0:1, :].reshape(-1, self.d_hidden)
        return hs[:, 1 + self._n_reg:, :].reshape(-1, self.d_hidden)


class CLIPRunner(BackboneRunner):
    def __init__(self, device: torch.device, clip_model=None, clip_processor=None):
        from transformers import CLIPModel, CLIPImageProcessor
        self.device = device
        if clip_model is not None:
            self._model = clip_model
        else:
            print(f"Loading CLIP vision backbone on {device}...")
            self._model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float32,
            ).to(device).eval()
        if clip_processor is not None:
            self._processor = clip_processor
        else:
            self._processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        self.d_hidden = self._model.config.vision_config.hidden_size

    def get_patch_tokens(self, img: Image.Image, image_size: int, token_type: str) -> torch.Tensor:
        # image_size and token_type are ignored — CLIP always uses 224 px spatial patches
        pv  = self._processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        out = self._model.vision_model(pixel_values=pv)
        # layout: [CLS, patch_0 … patch_255] — no register tokens
        return out.last_hidden_state[:, 1:, :].reshape(-1, self.d_hidden)


def make_runner(
    backbone: str,
    device: torch.device,
    clip_model=None,
    clip_processor=None,
) -> BackboneRunner:
    """
    Factory: return the appropriate BackboneRunner.

    Pass clip_model / clip_processor to reuse an already-loaded CLIP instance.
    """
    if backbone == 'clip':
        return CLIPRunner(device, clip_model=clip_model, clip_processor=clip_processor)
    if backbone.startswith('dinov2'):
        layer = int(backbone.split('_layer')[1]) if '_layer' in backbone else None
        return DINOv2Runner(device, layer=layer)
    return DINOv3Runner(device)


def load_batched_backbone(backbone_name: str, layer, device: torch.device):
    """
    Load a vision backbone for batched precompute scripts.

    Returns
    -------
    forward_fn   : callable (batch_tensor) -> (bs, n_tokens, d_hidden)
    d_hidden     : int
    n_reg        : int  (register tokens; 0 for CLIP)
    transform_fn : callable PIL Image -> Tensor (C, H, W)
    """
    use_intermediate = layer is not None

    if backbone_name == 'clip':
        from transformers import CLIPModel, CLIPImageProcessor
        print(f"Loading CLIP ViT-L/14 on {device}...")
        proc  = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float32,
        ).to(device).eval()
        d_hidden = model.config.vision_config.hidden_size  # 1024
        n_reg    = 0

        def transform_fn(img):
            return proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        if use_intermediate:
            _embed   = model.vision_model.embeddings
            _prenorm = model.vision_model.pre_layrnorm
            _layers  = model.vision_model.encoder.layers

            def forward_fn(imgs):
                h = _embed(imgs)
                h = _prenorm(h)
                for i, enc_layer in enumerate(_layers):
                    out = enc_layer(
                        hidden_states=h, attention_mask=None, causal_attention_mask=None,
                    )
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    if i == layer - 1:
                        return h
                return h
        else:
            def forward_fn(imgs):
                return model.vision_model(pixel_values=imgs).last_hidden_state

    elif backbone_name.startswith('dinov2'):
        print(f"Loading DINOv2 ViT-B/14 on {device}...")
        model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        ).to(device).eval()
        d_hidden = 768
        n_reg    = 0

        _dino_xfm = _dino_transform(224)

        def transform_fn(img):
            return _dino_xfm(img)

        if use_intermediate:
            _act = [None]
            def _hook(module, input, output):
                _act[0] = output
            model.blocks[layer].register_forward_hook(_hook)

            def forward_fn(imgs):
                _act[0] = None
                model.forward_features(imgs)
                return _act[0]  # (bs, 1 + n_patches, 768)
        else:
            def forward_fn(imgs):
                out = model.forward_features(imgs)
                cls     = out['x_norm_clstoken'].unsqueeze(1)
                patches = out['x_norm_patchtokens']
                return torch.cat([cls, patches], dim=1)

    else:  # dinov3
        from transformers import AutoModel
        print(f"Loading DINOv3 ViT-L/16 on {device}...")
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.float32,
        ).to(device).eval()
        d_hidden = model.config.hidden_size
        n_reg    = model.config.num_register_tokens

        _dino_xfm = _dino_transform(256)

        def transform_fn(img):
            return _dino_xfm(img)

        def forward_fn(imgs):
            out = model(pixel_values=imgs, output_hidden_states=use_intermediate)
            return out.hidden_states[layer] if use_intermediate else out.last_hidden_state

    layer_desc = f"layer {layer}" if use_intermediate else "final layer"
    print(f"  d_hidden={d_hidden}, register_tokens={n_reg}, extracting from {layer_desc}")
    return forward_fn, d_hidden, n_reg, transform_fn
