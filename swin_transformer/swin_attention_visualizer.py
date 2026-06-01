#!/usr/bin/env python3
"""
Swin Transformer Attention Map Visualizer
==========================================
Loads RGB images from a directory, runs them through a pretrained
swin_base_patch4_window12_384 model, extracts attention weights from
every WindowAttention block via forward hooks, and saves one figure
per block per image.

Optionally accepts a ground-truth mask directory.  For each image that has
a paired mask the script also writes a "perfect attention" figure showing
what the ideal attention map would look like — useful for direct comparison
with the model's actual attention.

Figure format (both predicted and perfect):
    [ Input image ]  |  [ Attention heatmap + colorbar ]
              "Block XX <label> (mean)"

Input images   : any size (512x512 recommended; resized to 384x384)
Mask images    : binary segmentation masks — non-black pixels = foreground.
                 Paired by filename stem after optional prefix substitution.
                 Default: replace "image_" with "mask_" in the image stem.
                 Override with --mask_prefix_from / --mask_prefix_to.
Model          : swin_base_patch4_window12_384  (pretrained ImageNet-22K->1K)
Output layout  :
    <output_dir>/<image_stem>/block_<NN>.png          <- model attention
    <output_dir>/<image_stem>/block_<NN>_perfect.png  <- ground-truth perfect
"""

import math
import argparse
from pathlib import Path

import torch
import timm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from timm.models.swin_transformer import WindowAttention

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "./images"
DEFAULT_OUTPUT_DIR = "./attention_maps"
MODEL_NAME         = "swin_base_patch4_window12_384"
IMG_SIZE           = 384
PATCH_SIZE         = 4
WINDOW_SIZE        = 12
COLORMAP           = "magma"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature-map size (patches) per stage for IMG_SIZE=384, PATCH_SIZE=4
# Stage 0: 96  Stage 1: 48  Stage 2: 24  Stage 3: 12
STAGE_FEAT_SIZES = {0: 96, 1: 48, 2: 24, 3: 12}
SUPPORTED_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── Mask configuration ────────────────────────────────────────────────────────
# Mask files are paired to images by transforming the image filename stem.
#   Default: "image_borebreen_1_0_0" -> "mask_borebreen_1_0_0"
# Set --mask_prefix_from="" --mask_prefix_to="" to match by identical stem.
DEFAULT_MASK_DIR         = None      # None = skip ground-truth masks
DEFAULT_MASK_PREFIX_FROM = ""  # prefix to strip from the image stem (empty = match by identical stem)
DEFAULT_MASK_PREFIX_TO   = ""  # replacement prefix for the mask stem (empty = match by identical stem)
# Foreground detection: pixel is foreground when max(R,G,B) > threshold (0-255)
MASK_FG_THRESHOLD = 30

# ── Image loading ─────────────────────────────────────────────────────────────
def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def load_image_paths(input_dir):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {p}")
    paths = sorted([f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
    if not paths:
        raise RuntimeError(f"No supported images found in {p}")
    return paths

# ── Model + hooks ─────────────────────────────────────────────────────────────
class AttentionStore:
    """
    Attaches forward hooks to the Softmax inside every WindowAttention block.

    Each record contains:
        key        : module path  e.g. "layers.0.blocks.1.attn"
        stage      : Swin stage index (0-3)
        block      : block index within stage
        global_idx : sequential 0-based index across all blocks
        attn       : Tensor (B*W, num_heads, N, N)
    """
    def __init__(self):
        self.records  = []
        self._handles = []
        self._counter = 0

    def _make_hook(self, key, stage, block):
        idx = self._counter
        self._counter += 1
        def hook(module, inp, out):
            self.records.append({
                "key":        key,
                "stage":      stage,
                "block":      block,
                "global_idx": idx,
                "attn":       out.detach().cpu().float(),
            })
        return hook

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, WindowAttention):
                parts = name.split(".")
                try:
                    stage = int(parts[1])
                    block = int(parts[3])
                except (IndexError, ValueError):
                    stage, block = -1, -1
                handle = module.softmax.register_forward_hook(
                    self._make_hook(name, stage, block)
                )
                self._handles.append(handle)
        return self

    def clear(self):
        self.records.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_model():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model.eval().to(DEVICE)


# ── Attention reconstruction ──────────────────────────────────────────────────
def reconstruct_spatial_map(attn, feat_h, feat_w, window_size):
    """
    Collapse (B*W, num_heads, N, N) into a normalised (feat_h, feat_w) map.

    1. Average over heads            -> (W, N, N)
    2. Average over keys (axis=-1)   -> (W, N)   per-token score
    3. Reshape into window grids     -> (W, wh, ww)
    4. Tile back to full feature map -> (feat_h, feat_w)
    5. Min-max normalise             -> [0, 1]
    """
    num_win_h   = max(1, feat_h // window_size)
    num_win_w   = max(1, feat_w  // window_size)
    num_windows = num_win_h * num_win_w

    available = attn.shape[0]
    if available < num_windows:
        num_windows = available
        side = int(math.sqrt(num_windows))
        num_win_h = num_win_w = side

    attn_mean    = attn[:num_windows].mean(dim=1)        # (W, N, N)
    token_scores = attn_mean.mean(dim=-1)                # (W, N)
    token_map    = token_scores.reshape(num_windows, window_size, window_size)

    spatial = (
        token_map
        .reshape(num_win_h, num_win_w, window_size, window_size)
        .permute(0, 2, 1, 3)
        .reshape(num_win_h * window_size, num_win_w * window_size)
        .numpy()
    )

    lo, hi = spatial.min(), spatial.max()
    if hi - lo > 1e-8:
        spatial = (spatial - lo) / (hi - lo)
    return spatial.astype(np.float32)


def upsample_to(arr, h, w):
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    return np.array(pil.resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0



# ── Mask loading & conversion ─────────────────────────────────────────────────
def find_mask_path(img_path, mask_dir, prefix_from, prefix_to, suffix=""):
    """
    Locate the mask file paired to *img_path*, trying three strategies:

    1. Prefix substitution : strip *prefix_from* from the image stem and
       prepend *prefix_to*   e.g. "image_foo" -> "mask_foo"
    2. Suffix append       : append *suffix* to the image stem
       e.g. "foo" -> "foo_mask"  (skipped when suffix is empty)
    3. Identical stem      : use the image stem unchanged (always tried last)

    Returns
    -------
    tuple (Path | None, str)
        Matched path (or None) and a human-readable string listing every
        stem pattern that was searched, for diagnostic output.
    """
    mask_dir_p = Path(mask_dir)
    stem = img_path.stem

    candidates = []

    # 1. prefix substitution
    if prefix_from and stem.startswith(prefix_from):
        candidates.append(prefix_to + stem[len(prefix_from):])
    elif prefix_to:
        candidates.append(prefix_to + stem)   # prepend even if no match

    # 2. suffix
    if suffix:
        candidates.append(stem + suffix)

    # 3. identical stem fallback
    candidates.append(stem)

    # deduplicate, preserve order
    seen, unique = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c); unique.append(c)

    tried_label = ", ".join(f'"{c}.*"' for c in unique)

    for mask_stem in unique:
        for ext in SUPPORTED_EXTS:
            candidate = mask_dir_p / (mask_stem + ext)
            if candidate.exists():
                return candidate, tried_label

    return None, tried_label


def load_mask_as_heatmap(mask_path, img_size=IMG_SIZE):
    """
    Load a binary segmentation mask and return a float32 heatmap
    of shape (img_size, img_size) with values in [0, 1].

    Foreground detection
    --------------------
    Any pixel where max(R, G, B) > MASK_FG_THRESHOLD is foreground = 1.0.
    The dark-red-on-black masks in the reference images satisfy this for
    any red pixel with R > 30.

    A mild Gaussian blur is applied so the "perfect" heatmap has smooth
    gradients that are visually comparable to the predicted attention maps.
    """
    from PIL import ImageFilter

    pil  = Image.open(mask_path).convert("RGB")
    pil  = pil.resize((img_size, img_size), resample=Image.NEAREST)
    arr  = np.array(pil)                              # (H, W, 3) uint8

    # binary foreground mask
    fg   = (arr.max(axis=-1) > MASK_FG_THRESHOLD).astype(np.float32)

    # mild blur so edges match the smooth look of attention heatmaps
    fg_pil  = Image.fromarray((fg * 255).astype(np.uint8))
    blurred = np.array(fg_pil.filter(ImageFilter.GaussianBlur(radius=6)))
    heatmap = blurred.astype(np.float32) / 255.0

    lo, hi = heatmap.min(), heatmap.max()
    if hi - lo > 1e-8:
        heatmap = (heatmap - lo) / (hi - lo)
    return heatmap


# ── Figure generation ─────────────────────────────────────────────────────────
def save_block_figure(original_pil, attn_map, block_title, output_path,
                      img_size=IMG_SIZE, cmap=COLORMAP):
    """
    Save one side-by-side figure matching the reference layout:

        ┌─────────────┬──────────────────────────┬───┐
        │ Input image │   Attention heatmap       │ c │
        │             │   (block_title)           │ b │
        └─────────────┴──────────────────────────┴───┘
    """
    orig_arr = np.array(
        original_pil.resize((img_size, img_size), resample=Image.BICUBIC)
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5),
        facecolor="white",
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
    )

    # Left panel: input image
    axes[0].imshow(orig_arr)
    axes[0].set_title("Input image", fontsize=12, pad=6, color="black")
    axes[0].axis("off")

    # Right panel: attention heatmap
    im = axes[1].imshow(attn_map, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(block_title, fontsize=12, pad=6, color="black")
    axes[1].axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02,
                        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=9, colors="black")
    cbar.outline.set_edgecolor("black")

    fig.subplots_adjust(left=0.02, right=0.93, top=0.92, bottom=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(input_dir, output_dir,
        mask_dir=None, mask_prefix_from=DEFAULT_MASK_PREFIX_FROM,
        mask_prefix_to=DEFAULT_MASK_PREFIX_TO, mask_suffix=""):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Device    : {DEVICE}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Input     : {input_dir}")
    print(f"Output    : {out_root}")
    if mask_dir:
        print(f"Masks     : {mask_dir}  "
              f"(stem: '{mask_prefix_from}...' -> '{mask_prefix_to}...')")
    print()

    # ── model ─────────────────────────────────────────────────────────────
    print("Loading pretrained model ...")
    model = build_model()

    store = AttentionStore()
    store.register(model)
    print(f"Registered {len(store._handles)} attention hooks.\n")

    tf          = build_transform(IMG_SIZE)
    image_paths = load_image_paths(input_dir)
    print(f"Found {len(image_paths)} image(s).\n")

    for img_path in image_paths:
        print(f"Processing : {img_path.name}")

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"  [skip] {exc}")
            continue

        # ── optional mask ──────────────────────────────────────────────
        perfect_heatmap = None
        if mask_dir:
            mpath, tried = find_mask_path(img_path, mask_dir,
                                          mask_prefix_from, mask_prefix_to,
                                          mask_suffix)
            if mpath:
                perfect_heatmap = load_mask_as_heatmap(mpath, IMG_SIZE)
                print(f"  Mask      : {mpath.name}")
            else:
                print(f"  Mask      : [not found] searched in {mask_dir} for {tried}")

        # ── forward pass ───────────────────────────────────────────────
        tensor = tf(pil_img).unsqueeze(0).to(DEVICE)
        store.clear()
        with torch.no_grad():
            _ = model(tensor)

        n_blocks    = len(store.records)
        img_out_dir = out_root / img_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Blocks    : {n_blocks}  ->  {img_out_dir}/")

        for rec in store.records:
            feat_hw = STAGE_FEAT_SIZES.get(rec["stage"], IMG_SIZE // PATCH_SIZE)
            bidx    = rec["global_idx"]

            # ── predicted attention ────────────────────────────────────
            smap    = reconstruct_spatial_map(
                          rec["attn"], feat_hw, feat_hw, WINDOW_SIZE)
            smap_up = upsample_to(smap, IMG_SIZE, IMG_SIZE)
            save_block_figure(
                pil_img, smap_up,
                block_title=f"Block {bidx:02d} attention (mean)",
                output_path=img_out_dir / f"block_{bidx:02d}.png",
            )

            # ── perfect attention (ground truth) ───────────────────────
            if perfect_heatmap is not None:
                save_block_figure(
                    pil_img, perfect_heatmap,
                    block_title=f"Block {bidx:02d} perfect attention (ground truth)",
                    output_path=img_out_dir / f"block_{bidx:02d}_perfect.png",
                )

        # print block list
        names = [f"block_{r['global_idx']:02d}.png" for r in store.records]
        if perfect_heatmap is not None:
            perf  = [f"block_{r['global_idx']:02d}_perfect.png"
                     for r in store.records]
            names = [f for pair in zip(names, perf) for f in pair]
        for n in names:
            print(f"    + {n}")
        print()

    store.remove_hooks()
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Visualise Swin Transformer per-block attention maps. "
            "Optionally generate ground-truth 'perfect' attention figures "
            "from binary segmentation masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  "-i",
                   default=DEFAULT_INPUT_DIR,
                   help="Directory of input RGB images.")
    p.add_argument("--output_dir", "-o",
                   default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory. "
                        "One sub-folder per image is created.")
    p.add_argument("--mask_dir",   "-m",
                   default=DEFAULT_MASK_DIR,
                   help="Directory of binary segmentation masks. "
                        "If omitted, perfect-attention figures are skipped.")
    p.add_argument("--mask_prefix_from",
                   default=DEFAULT_MASK_PREFIX_FROM,
                   help="Prefix to strip from the image stem when locating "
                        "its paired mask  (e.g. 'image_').")
    p.add_argument("--mask_prefix_to",
                   default=DEFAULT_MASK_PREFIX_TO,
                   help="Replacement prefix for the mask stem "
                        "(e.g. 'mask_').")
    p.add_argument("--mask_suffix",
                   default="",
                   help="Suffix appended to the image stem to find the mask "
                        "(e.g. '_mask' matches 'foo_mask.png'). "
                        "Used in addition to prefix substitution.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        mask_dir         = args.mask_dir,
        mask_prefix_from = args.mask_prefix_from,
        mask_prefix_to   = args.mask_prefix_to,
        mask_suffix      = args.mask_suffix,
    )
