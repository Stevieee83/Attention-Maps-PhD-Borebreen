import os
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from load_data import LoadData

import argparse

COLORMAP = "magma"

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--model_type", type=str, default='ViT-L-Sat')
parser.add_argument("--images_no", type=int, default=10)
parser.add_argument("--image_size", type=int, default=512)
parser.add_argument("--image_dir", type=str, default="/mnt/storage/Massachusetts Buildings Dataset/test")
parser.add_argument("--labels_dir", type=str, default="/mnt/storage/Massachusetts Buildings Dataset/test_labels")
parser.add_argument("--weights", type=str, default="/mnt/storage/Linux-Desktop-8TB/VSCode/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
parser.add_argument("--output_dir", type=str, default="/mnt/storage/PhD-results/dinov3-vit-l-output_attn_maps-massachusetts-Buildings/512/train/")
# ------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    load_data = LoadData(args.image_dir, args.labels_dir)

    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    ##############################################################################################
    # Load the DINOv3 model backbone and send to the CUDA device
    # examples of available DINOv3 models:
    MODEL_DINOV3_VITS = "dinov3_vits16"
    MODEL_DINOV3_VITSP = "dinov3_vits16plus"
    MODEL_DINOV3_VITB = "dinov3_vitb16"
    MODEL_DINOV3_VITL = "dinov3_vitl16"
    MODEL_DINOV3_VITHP = "dinov3_vith16plus"
    MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

    MODEL_NAME = MODEL_DINOV3_VITL

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        weights=args.weights,
    )

    model.cuda()
    model.eval()
    print(model)

    print_model_parameters(model, model_name=MODEL_NAME)

    ##############################################################################################
    MODEL_TO_NUM_LAYERS = {
        MODEL_DINOV3_VITS: 12,
        MODEL_DINOV3_VITSP: 12,
        MODEL_DINOV3_VITB: 12,
        MODEL_DINOV3_VITL: 24,
        MODEL_DINOV3_VITHP: 32,
        MODEL_DINOV3_VIT7B: 40,
    }

    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

    # Number of register tokens (DINOv3 may include these between CLS and patch tokens)
    n_registers = getattr(model, 'num_register_tokens', 0)

    ##############################################################################################
    # Register forward hooks on each transformer block's attention module.
    # Each hook recomputes Q @ K^T from the QKV projection to capture the softmax
    # attention weights without modifying the model forward pass.
    stored_attentions = [None] * n_layers

    def make_attention_hook(layer_idx):
        def hook(module, input, output):
            if not hasattr(module, 'qkv'):
                return
            x = input[0]
            B, N, C = x.shape
            head_dim = C // module.num_heads
            scale = getattr(module, 'scale', head_dim ** -0.5)
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            stored_attentions[layer_idx] = attn.detach().cpu()
        return hook

    hooks = [block.attn.register_forward_hook(make_attention_hook(i))
             for i, block in enumerate(model.blocks)]

    ##############################################################################################
    PATCH_SIZE = 16
    IMAGE_SIZE = args.image_size
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def resize_transform(
        image: Image,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
    ) -> torch.Tensor:
        w, h = image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))

    ##############################################################################################
    images, labels = load_data.sequence_data_loading()
    n_images = min(len(images), args.images_no)

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Processing images"):
                image_i = images[i].convert('RGB')
                image_tensor = resize_transform(image_i)

                img_h = image_tensor.shape[1]
                img_w = image_tensor.shape[2]
                h_patches = img_h // PATCH_SIZE
                w_patches = img_w // PATCH_SIZE

                image_normalized = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_cuda = image_normalized.unsqueeze(0).cuda()

                # Forward pass — triggers the registered attention hooks
                _ = model(image_cuda)

                image_output_dir = Path(args.output_dir) / f"image_{i:04d}"
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # Unnormalized image as numpy array for plotting
                img_np = image_tensor.permute(1, 2, 0).numpy().clip(0, 1)

                for block_idx in range(n_layers):
                    attn = stored_attentions[block_idx]
                    if attn is None:
                        continue

                    n_heads = attn.shape[1]
                    patch_start = 1 + n_registers
                    n_patch_tokens = h_patches * w_patches

                    # CLS-to-patch attention averaged across heads: [h_patches, w_patches]
                    cls_attn = attn[0, :, 0, patch_start:patch_start + n_patch_tokens]
                    mean_attn = cls_attn.reshape(n_heads, h_patches, w_patches).numpy().mean(axis=0)

                    # Normalise to [0, 1] then upsample to image pixel dimensions
                    lo, hi = mean_attn.min(), mean_attn.max()
                    if hi - lo > 1e-8:
                        mean_attn = (mean_attn - lo) / (hi - lo)
                    mean_attn_up = upsample_to(mean_attn, img_h, img_w)

                    save_block_figure(
                        img_np, mean_attn_up,
                        block_title=f"Block {block_idx:02d} attention (mean)",
                        output_path=image_output_dir / f"block_{block_idx:02d}.png",
                    )

                print(f"Saved attention maps for image {i} to {image_output_dir}")

    for hook in hooks:
        hook.remove()

    print("DINOv3 Attention Map Script Complete")


def save_block_figure(img_np, attn_map, block_title, output_path, cmap=COLORMAP):
    """
    Save one side-by-side figure:
        [ Input image ] | [ Attention heatmap + colorbar ]
    Matches the layout used by swin_attention_visualizer.py.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11, 5),
        facecolor="white",
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Input image", fontsize=12, pad=6, color="black")
    axes[0].axis("off")

    im = axes[1].imshow(attn_map, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(block_title, fontsize=12, pad=6, color="black")
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02,
                        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=9, colors="black")
    cbar.outline.set_edgecolor("black")

    fig.subplots_adjust(left=0.02, right=0.93, top=0.92, bottom=0.04)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def upsample_to(arr, h, w):
    """Upsample a 2D float [0, 1] array to (h, w) via bilinear interpolation"""
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    return np.array(pil.resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0


def print_model_parameters(model, model_name="DINOv3"):
    """Print a summary of model parameter counts and estimated size"""
    print(f"\n{'='*60}")
    print(f"  {model_name} MODEL PARAMETERS")
    print(f"{'='*60}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable params:  {total_params - trainable_params:,}")
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Estimated size (MB):   {model_size_mb:.2f}")
    print(f"{'='*60}\n")


# Executes the main method from the DINOv3_attn_maps.py Python script
if __name__ == '__main__':
    main()
