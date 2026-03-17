import torch
from torchvision import transforms
from torchvision.transforms import v2
from overcomplete.models import BaseModel


class DinoV2(BaseModel):
    """
    DINOv2 ViT-B/14 with intermediate layer extraction.

    Token layout: [CLS, patch_0 … patch_N] — no register tokens.
    d_hidden = 768, patch_size = 14 → 16×16 patches for 224×224 images.

    Parameters
    ----------
    use_half : bool
    device : str
    extract_layers : list of int
        0-indexed block indices to extract (0–11 for ViT-B/14).
    """

    NUM_REGISTER_TOKENS = 0

    def __init__(self, use_half=False, device='cpu', extract_layers=None):
        super().__init__(use_half, device)

        self.model = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        ).eval().to(self.device)

        if self.use_half:
            self.model = self.model.half()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.extract_layers = extract_layers if extract_layers is not None else []
        self._activations = {}

        for layer_idx in self.extract_layers:
            if 0 <= layer_idx < len(self.model.blocks):
                self.model.blocks[layer_idx].register_forward_hook(self._get_hook(layer_idx))
            else:
                print(f"Warning: Layer {layer_idx} requested but model only has {len(self.model.blocks)} blocks.")

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self._activations[layer_idx] = output
        return hook

    def forward_features(self, x):
        """
        Returns dict of layer_idx → Tensor (batch, tokens, 768).
        'final' key holds the normalized patch tokens from the last layer.
        """
        self._activations = {}
        with torch.no_grad():
            if self.use_half:
                x = x.half()
            out = self.model.forward_features(x)
            results = self._activations.copy()
            results['final'] = out['x_patchtokens']
        return results


class DinoV3(BaseModel):
    """
    Concrete class for DINOv3 model with multi-layer extraction capabilities.

    DINOv3 uses ViT-B/16 (patch size 16), producing a 14x14 spatial grid
    for 224x224 images. Tokens: 1 CLS + 4 register tokens + 196 patch tokens.

    Parameters
    ----------
    use_half : bool, optional
        Whether to use half-precision (float16), by default False.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda'), by default 'cpu'.
    extract_layers : list of int, optional
        List of block indices (0-indexed) to extract features from.
        e.g., [2, 5, 8, 11].
    """

    NUM_REGISTER_TOKENS = 4  # DINOv3 uses 4 register tokens

    def __init__(self, use_half=False, device='cpu', extract_layers=None):
        super().__init__(use_half, device)

        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(
            "facebook/dinov3-vitb16-pretrain-lvd1689m"
        ).eval().to(self.device)

        if self.use_half:
            self.model = self.model.half()

        # Preprocessing (Standard DINOv3 - same ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- Hook Setup ---
        self.extract_layers = extract_layers if extract_layers is not None else []
        self._activations = {}

        if self.extract_layers:
            encoder = self.model.encoder.layer
            for layer_idx in self.extract_layers:
                if 0 <= layer_idx < len(encoder):
                    encoder[layer_idx].register_forward_hook(self._get_hook(layer_idx))
                else:
                    print(f"Warning: Layer {layer_idx} requested but model only has {len(encoder)} blocks.")

    def _get_hook(self, layer_idx):
        """Creates a closure to save the output of a specific layer."""
        def hook(module, input, output):
            # HuggingFace ViT block output is a tuple; first element is hidden state
            self._activations[layer_idx] = output[0]
        return hook

    def forward_features(self, x):
        """
        Perform a forward pass and extract intermediate features.

        Returns
        -------
        dict
            Keys are layer indices (int).
            Values are torch.Tensor of shape (batch, tokens, dim).
            Tokens include CLS + register tokens + patch tokens.
        """
        self._activations = {}

        with torch.no_grad():
            if self.use_half:
                x = x.half()

            outputs = self.model(pixel_values=x)

            results = self._activations.copy()
            results['final'] = outputs.last_hidden_state

            return results