"""Pure HTML-emitting helpers used by the various Bokeh ``Div`` widgets.

No Bokeh, no ``state``, no I/O — just string formatting plus image
encoding (via :mod:`explorer.images`). Tightly state-coupled HTML helpers
(``_make_summary_html``, ``_feature_block_html``, the cross-SAE
comparison) live alongside the things they read in the panel modules /
:mod:`explorer.rendering` once Context lands.
"""

from . import images as _images
# `_render_overlay_from_ds` is imported lazily inside the cross-SAE function
# to avoid a hard cycle (rendering imports runtime, runtime imports images,
# images is imported here too — keeping the rendering import deferred lets
# this module stay importable from anywhere).


# Status banner styles; first arg is a state-name key (NOT the runtime
# ``_State`` instance — this function is pure).
_STATUS_STYLES = {
    'idle':    'background:#f5f5f5;border-left:4px solid #bbb;color:#666',
    'loading': 'background:#fff8e0;border-left:4px solid #f0a020;color:#7a5000',
    'ok':      'background:#e8f4e8;border-left:4px solid #2a8a2a;color:#1a5a1a',
    'dead':    'background:#fce8e8;border-left:4px solid #c03030;color:#8a1a1a',
}


def _status_html(state: str, msg: str) -> str:
    """Return a styled HTML status banner. ``state`` is one of
    ``idle`` / ``loading`` / ``ok`` / ``dead``."""
    style = _STATUS_STYLES.get(state, _STATUS_STYLES['idle'])
    return (f'<div style="{style};padding:7px 12px;border-radius:3px;'
            f'font-size:13px">{msg}</div>')


def make_image_grid_html(images_info, title: str) -> str:
    """Render a flex grid of (PIL, caption) tuples under a title heading."""
    if not images_info:
        return (f'<h3 style="margin:4px 0 6px 0;color:#444;border-bottom:2px solid #e8e8e8;'
                f'padding-bottom:4px">{title}</h3>'
                f'<p style="color:#aaa;font-style:italic;margin:4px 0">No examples available</p>')
    thumb_w = min(_images.THUMB, 224)
    html = (f'<h3 style="margin:4px 0 8px 0;color:#333;border-bottom:2px solid #e0e0e0;'
            f'padding-bottom:4px">{title}</h3>')
    html += '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:2px 0 10px 0">'
    for img, caption in images_info:
        url = _images.pil_to_data_url(img)
        parts = caption.split('<br>')
        cap_html = ''.join(f'<div>{p}</div>' for p in parts)
        html += (f'<div style="text-align:center;width:{thumb_w}px">'
                 f'<img src="{url}" width="{thumb_w}" height="{thumb_w}"'
                 f' style="border:1px solid #d0d0d0;border-radius:5px;display:block"/>'
                 f'<div style="font-size:10px;color:#555;margin-top:3px;line-height:1.4">'
                 f'{cap_html}</div></div>')
    html += '</div>'
    return html


def make_compare_aggregations_html(top_infos, mean_infos, p75_infos, feat: int,
                                   n_each: int = 6, model_label: str = None) -> str:
    """Figure-ready side-by-side comparison of the first two aggregation methods.

    Only Top (Max Activation) and Mean Activation are shown so that a
    screenshot of this element stands alone as a clean figure panel.
    """
    col_thumb = min(_images.THUMB, 160)

    # Only the first two methods are shown in the figure
    sections = [
        ("Top Activation",  "#2563a8", top_infos),
        ("Mean Activation", "#1a7a4a", mean_infos),
    ]

    cols_per_row = 2
    strip_w = cols_per_row * col_thumb + (cols_per_row - 1) * 6

    # Outer container — white background, no border decoration so the
    # figure can be cropped cleanly. A subtle bottom-padding keeps images
    # from being clipped.
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;background:#ffffff;'
        'padding:16px 20px 14px 20px;display:inline-block">'
        f'<div style="font-size:13px;font-weight:bold;color:#222;margin-bottom:14px;'
        f'letter-spacing:0.1px">'
        + (f'{model_label} — ' if model_label else '')
        + f'Feature {feat}</div>'
        '<div style="display:flex;gap:24px;align-items:flex-start">'
    )

    for method_name, color, infos in sections:
        shown = (infos or [])[:n_each]

        html += (
            f'<div style="display:inline-flex;flex-direction:column">'
            f'<div style="background:{color};color:#ffffff;font-size:13px;font-weight:bold;'
            f'text-align:center;padding:6px 0;border-radius:5px;margin-bottom:10px;'
            f'letter-spacing:0.4px;width:{strip_w}px;box-sizing:border-box">{method_name}</div>'
            f'<div style="display:grid;grid-template-columns:repeat({cols_per_row},{col_thumb}px);gap:6px">'
        )
        if not shown:
            html += '<div style="color:#aaa;font-style:italic;font-size:11px;padding:8px">No images</div>'
        for img, caption in shown:
            url = _images.pil_to_data_url(img)
            parts = caption.split('<br>')
            cap_html = '<br>'.join(parts)
            html += (
                f'<div style="text-align:center">'
                f'<img src="{url}" width="{col_thumb}" height="{col_thumb}"'
                f' style="border:1px solid #ccc;border-radius:3px;display:block"/>'
                f'<div style="font-size:9px;color:#555;margin-top:3px;line-height:1.35">'
                f'{cap_html}</div></div>'
            )
        html += '</div></div>'

    html += '</div></div>'
    return html


def make_cross_sae_comparison_html(ds_a: dict, feat_a: int, ds_b: dict, feat_b: int,
                                   n: int = 4, size: int = 160, alpha: float = 1.0) -> str:
    """Two side-by-side 2x2 grids: left = SAE A / feat_a, right = SAE B / feat_b.

    Reads top images + heatmaps from the two dataset dicts directly, so it
    works across datasets without needing the active state.
    """
    from .rendering import _render_overlay_from_ds  # lazy import (see top)

    def _collect(ds, feat):
        items = []
        for slot in range(min(n, ds['top_img_idx'].shape[1])):
            result = _render_overlay_from_ds(ds, feat, slot, size=size, alpha=alpha)
            if result:
                items.append(result)
            if len(items) == n:
                break
        return items

    items_a = _collect(ds_a, feat_a)
    items_b = _collect(ds_b, feat_b)

    def _strip_dim(label: str) -> str:
        """Strip parenthetical dim info like '(d=32K)' or '(d=32K, k=160)'."""
        out = label
        while '(' in out and ')' in out:
            l, r = out.index('('), out.index(')')
            out = out[:l].rstrip() + out[r + 1:]
        return out.strip(' —').strip()

    def _grid_html(items, model_label: str, feat_num: int, color: str) -> str:
        header = (
            f'<div style="background:{color};color:#fff;text-align:center;'
            f'padding:5px 6px 4px 6px;border-radius:4px;margin-bottom:6px;line-height:1.4">'
            f'<div style="font-size:12px;font-weight:bold">{model_label}</div>'
            f'<div style="font-size:10px;opacity:0.88">Feature {feat_num}</div>'
            f'</div>'
        )
        grid = '<div style="display:grid;grid-template-columns:repeat(2,{s}px);gap:4px">'.format(s=size)
        for img, cap in items:
            url = _images.pil_to_data_url(img)
            grid += (f'<div style="text-align:center">'
                     f'<img src="{url}" width="{size}" height="{size}"'
                     f' style="border:1px solid #ccc;border-radius:3px;display:block"/>'
                     f'<div style="font-size:9px;color:#555;margin-top:2px">{cap}</div></div>')
        grid += '</div>'
        return f'<div style="display:flex;flex-direction:column">{header}{grid}</div>'

    label_a = _strip_dim(ds_a['label'])
    label_b = _strip_dim(ds_b['label'])
    col_a = _grid_html(items_a, label_a, feat_a, "#2563a8")
    col_b = _grid_html(items_b, label_b, feat_b, "#b85c00")

    return (
        '<div style="display:flex;gap:16px;padding:8px;background:#fafafa;'
        'border:1px solid #ddd;border-radius:6px">'
        + col_a + col_b + '</div>'
    )
