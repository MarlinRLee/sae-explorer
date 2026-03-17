"""
CLIP text-alignment utilities for SAE feature interpretation.

Key functions:
- compute_text_embeddings: encode text strings into L2-normalised CLIP embeddings.
- compute_mei_text_alignment: align SAE features to text via their top MEI images.
- compute_text_alignment: dot-product similarity between precomputed feature/text embeds.
- search_features_by_text: find top-k features for a free-text query.

The precomputed scores can be stored in explorer_data.pt under:
    'clip_text_scores'   : Tensor (n_features, n_vocab)  float16
    'clip_text_vocab'    : list[str]
    'clip_feature_embeds': Tensor (n_features, clip_proj_dim)  float32
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_clip(device: str | torch.device = "cpu", model_name: str = "openai/clip-vit-large-patch14"):
    """
    Load a CLIP model and processor.

    Parameters
    ----------
    device : str or torch.device
    model_name : str
        HuggingFace model ID.  Default matches the ViT-L/14 variant used by
        many vision papers and is a reasonable match for DINOv3-ViT-L/16.

    Returns
    -------
    model : CLIPModel (eval mode, on device)
    processor : CLIPProcessor
    """
    print(f"Loading CLIP ({model_name})...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()
    print(f"  CLIP loaded (d_text={model.config.projection_dim})")
    return model, processor


# ---------------------------------------------------------------------------
# Core alignment computation
# ---------------------------------------------------------------------------

def compute_text_embeddings(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str | torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode a list of text strings into L2-normalised CLIP text embeddings.

    Returns
    -------
    Tensor of shape (len(texts), clip_proj_dim), float32, on CPU.
    """
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            # Go through text_model + text_projection directly to avoid
            # version differences in get_text_features() return type.
            text_out = model.text_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
            )
            embeds = model.text_projection(text_out.pooler_output)
            embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu().float())
    return torch.cat(all_embeds, dim=0)  # (n_texts, clip_proj_dim)


def compute_text_alignment(
    feature_vision_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between feature embeddings and text
    embeddings.  Both inputs must already be L2-normalised.

    Parameters
    ----------
    feature_vision_embeds : Tensor (n_features, d)
    text_embeds : Tensor (n_texts, d)

    Returns
    -------
    Tensor (n_features, n_texts) of cosine similarities in [-1, 1].
    """
    return feature_vision_embeds @ text_embeds.T   # (n_features, n_texts)


# ---------------------------------------------------------------------------
# MEI-based text alignment (more accurate, more expensive)
# ---------------------------------------------------------------------------

def compute_mei_text_alignment(
    top_img_paths: list[list[str]],
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str | torch.device,
    n_top_images: int = 4,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    For each feature, compute the mean CLIP image embedding of its top-N MEIs,
    then return cosine similarity against each text embedding.

    This is the most principled approach: CLIP operates on actual images, so
    the alignment reflects the true visual concept captured by the feature.

    Parameters
    ----------
    top_img_paths : list of lists
        top_img_paths[i] = list of image file paths for feature i's MEIs.
    texts : list[str]
        Text queries / vocabulary concepts.
    n_top_images : int
        How many MEIs to average per feature.
    batch_size : int

    Returns
    -------
    Tensor (n_features, n_texts) float32, on CPU.
    """
    from PIL import Image

    n_features = len(top_img_paths)
    text_embeds = compute_text_embeddings(texts, model, processor, device)
    # text_embeds: (n_texts, d)

    feature_img_embeds = []
    for feat_paths in top_img_paths:
        paths = [p for p in feat_paths[:n_top_images] if p]
        if not paths:
            feature_img_embeds.append(torch.zeros(model.config.projection_dim))
            continue

        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=imgs, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        with torch.inference_mode():
            vision_out = model.vision_model(pixel_values=pixel_values)
            img_embeds = model.visual_projection(vision_out.pooler_output)  # (n_imgs, d)
            img_embeds = F.normalize(img_embeds, dim=-1)
            mean_embed = img_embeds.mean(dim=0)
            mean_embed = F.normalize(mean_embed, dim=-1)
        feature_img_embeds.append(mean_embed.cpu().float())

    feature_img_embeds = torch.stack(feature_img_embeds, dim=0)  # (n_feat, d)
    return feature_img_embeds @ text_embeds.T                     # (n_feat, n_texts)


# ---------------------------------------------------------------------------
# Feature search by free-text query
# ---------------------------------------------------------------------------

def search_features_by_text(
    query: str,
    clip_scores: torch.Tensor,
    vocab: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str | torch.device,
    top_k: int = 20,
    feature_embeds: torch.Tensor | None = None,
) -> list[tuple[int, float]]:
    """
    Find the top-k SAE features most aligned with a free-text query.

    If the query is already in `vocab`, use the precomputed scores directly.
    Otherwise encode the query on-the-fly and compute dot products against
    `feature_embeds` (the per-feature MEI image embeddings stored as
    'clip_feature_embeds' in explorer_data.pt).

    Parameters
    ----------
    query : str
    clip_scores : Tensor (n_features, n_vocab)
        Precomputed alignment matrix (L2-normalised features × L2-normalised
        text embeddings).
    vocab : list[str]
    model, processor, device : CLIP model components (used for on-the-fly encoding)
    top_k : int
    feature_embeds : Tensor (n_features, clip_proj_dim) or None
        L2-normalised per-feature MEI image embeddings.  Required for
        free-text queries that are not in `vocab`.

    Returns
    -------
    list of (feature_idx, score) sorted by score descending.
    """
    if query in vocab:
        col = vocab.index(query)
        scores_vec = clip_scores[:, col].float()                  # (n_features,)
    else:
        if feature_embeds is None:
            raise ValueError(
                "Free-text query requires 'feature_embeds' (clip_feature_embeds "
                "from explorer_data.pt).  Pass feature_embeds=data['clip_feature_embeds'] "
                "or restrict queries to vocab terms."
            )
        q_embed = compute_text_embeddings([query], model, processor, device)  # (1, d)
        scores_vec = (feature_embeds.float() @ q_embed.T).squeeze(-1)        # (n_features,)

    top_indices = torch.topk(scores_vec, k=min(top_k, len(scores_vec))).indices
    return [(int(i), float(scores_vec[i])) for i in top_indices]
