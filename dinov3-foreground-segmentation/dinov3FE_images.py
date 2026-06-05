import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import pandas as pd

from load_data import LoadData
from output_data_fe_npy_idx import OutputDataFENpyIdx

import wandb

import argparse

# Defines the ArgumentParser object
parser = argparse.ArgumentParser()

# Input parameters
parser.add_argument("--model_type", type=str, default='ViT-L-Sat')
parser.add_argument("--model_name", type=str, default="dinov3_vitl16", choices=['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16'], help="Model architecture to use")
parser.add_argument("--num_layers", type=int, default=24, choices=[12, 24, 32, 40], help="Number of layers (12: ViT-S/SP/B, 24: ViT-L, 32: ViT-HP, 40: ViT-7B)")
parser.add_argument("--images_no", type=int, default=24)
parser.add_argument("--image_size", type=int, default=512) # 768
parser.add_argument("--image_dir", type=str, default="/home/steven/Documents/VSCode/borebreen_image_data/512/train/images")
parser.add_argument("--labels_dir", type=str, default="/home/steven/Documents/VSCode/borebreen_image_data/512/train/masks")
parser.add_argument("--weights", type=str, default="/mnt/storage/Linux-Desktop-8TB/VSCode/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
parser.add_argument("--output_npy", type=str, default="./output_npy/512/train/")
parser.add_argument("--X", type=str, default='X_train.npy')
parser.add_argument("--y", type=str, default='y_train.npy')
parser.add_argument("--image_index", type=str, default='image_index.npy')
# ------------------------------------------------------------------------

# Main funciton to sequence the Python script source code
def main():
    # Creates the ArgumentParser object in the main function
    args = parser.parse_args()
    
    load_data = LoadData(args.image_dir, args.labels_dir)

    # Initialise the Weights and Biases run
    wandb.init(project=f"DINOv3 Segmentation FE to .npy {args.model_type} Images 512 Test",
                name=f"Images: {args.images_no}")

    # Copy your config
    config = wandb.config

    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    ##############################################################################################
    # Load the DINOv3 model backbone and send to the CUDA device
    MODEL_NAME = args.model_name

    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
        weights=args.weights,
    )

    model.cuda()
    print(model)

    print_model_parameters(model, model_name=MODEL_NAME)
    print_weight_values(model, model_name="DINOv3", max_layers=10, max_values_per_layer=20)

    ##############################################################################################
    # Load images and labels in sorted filename order so that image ID 0
    # always corresponds to the first alphabetically sorted file, consistent
    # with the sorted order used in inference_mode_images.py.
    _img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
    image_files_sorted = sorted(
        f for f in os.listdir(args.image_dir) if f.lower().endswith(_img_exts)
    )
    label_files_sorted = sorted(
        f for f in os.listdir(args.labels_dir) if f.lower().endswith(_img_exts)
    )

    images = []
    for fname in image_files_sorted:
        img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
        images.append(img)
    print(f"Loaded {len(images)} images (sorted order).")

    raw_labels = []
    for fname in label_files_sorted:
        lbl = Image.open(os.path.join(args.labels_dir, fname)).convert('L')
        lbl = lbl.resize((1024, 1024), Image.LANCZOS)
        raw_labels.append(lbl)
    print(f"Loaded {len(raw_labels)} labels (sorted order, resized to 1024x1024).")
    load_data.print_image_info(raw_labels)
    labels = load_data.binary_labels_convert(raw_labels)

    n_images = len(images)
    assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"

    ##############################################################################################
    # Building the Per-Patch Label Map
    PATCH_SIZE = 16
    IMAGE_SIZE = args.image_size

    # quantization filter for the given patch size
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    # image resize transform to dimensions divisible by patch size
    def resize_transform(
        mask_image: Image,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
)   -> torch.Tensor:
        w, h = mask_image.size
        h_patches = int(image_size / patch_size)
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
   
    ##############################################################################################
    xs = []
    ys = []
    image_index = []

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    n_layers = args.num_layers

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i in tqdm(range(n_images), desc="Processing images"):
                # Loading the ground truth
                mask_i = labels[i]#.split()[-1]
                mask_i_resized = resize_transform(mask_i)
                mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
                ys.append(mask_i_quantized)
                # Loading the image data 
                image_i = images[i].convert('RGB')
                image_i_resized = resize_transform(image_i)
                image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                image_i_resized = image_i_resized.unsqueeze(0).cuda()

                feats = model.get_intermediate_layers(image_i_resized, n=range(n_layers), reshape=True, norm=True)
                dim = feats[-1].shape[1]
                xs.append(feats[-1].squeeze().view(dim, -1).permute(1,0).detach().cpu())

                image_index.append(i * torch.ones(ys[-1].shape))


    # Concatenate all lists into torch tensors 
    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)

    print("Design full matrix of size : ", xs.shape)
    print("Label full matrix of size : ", ys.shape)
    print("Image full index matrix of size : ", image_index.shape)
    
    # Get unique values
    unique_values = torch.unique(ys)
    print("Unique label values:", unique_values)

    os.makedirs(args.output_npy, exist_ok=True)
    x_npy_path = args.output_npy + args.X
    y_npy_path = args.output_npy + args.y
    image_index_npy_path = args.output_npy + args.image_index

    # Define the OutputDataFENpyIdx Python object and load the X (features), y (labels) and image index
    output_data = OutputDataFENpyIdx(xs, ys, image_index, x_npy_path, y_npy_path, image_index_npy_path)

    # Call the Python class methods to output the X, y and image index data from the DINOv3 feature extractor
    output_data.tensor_to_npy_features()
    output_data.tensor_to_npy_labels()
    output_data.tensor_to_npy_image_index()
    print(f'Saved {args.X}, {args.y} and {args.image_index} to their output filepath directory')
    print("DINOv3 Feature Extractor Script Complete")

    # Close your Weights and biases run
    wandb.finish()

def print_model_parameters(model, model_name="DINOv3"):
    """Print detailed information about model parameters"""
    
    print(f"\n{'='*60}")
    print(f"  {model_name} MODEL PARAMETERS")
    print(f"{'='*60}")
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n{'Layer Name':<40} {'Shape':<25} {'Parameters':<15} {'Trainable'}")
    print("-" * 90)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            trainable_status = "Yes"
        else:
            trainable_status = "No"
        
        shape_str = str(list(param.shape))
        print(f"{name:<40} {shape_str:<25} {param_count:<15,} {trainable_status}")
    
    print("-" * 90)
    print(f"{'TOTAL PARAMETERS':<40} {'':<25} {total_params:<15,}")
    print(f"{'TRAINABLE PARAMETERS':<40} {'':<25} {trainable_params:<15,}")
    print(f"{'NON-TRAINABLE PARAMETERS':<40} {'':<25} {total_params - trainable_params:<15,}")
    
    # Calculate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes per param)
    print(f"{'ESTIMATED SIZE (MB)':<40} {'':<25} {model_size_mb:<15.2f}")
    print(f"{'='*60}")

def print_weight_values(model, model_name="DINOv3", max_layers=10, max_values_per_layer=20):
    """Print actual weight parameter values from the model"""
    
    print(f"\n{'='*80}")
    print(f"  {model_name} WEIGHT PARAMETER VALUES")
    print(f"{'='*80}")
    
    layer_count = 0
    
    for name, param in model.named_parameters():
        if layer_count >= max_layers:
            print(f"\n... (showing first {max_layers} layers only)")
            break
            
        print(f"\n{'-'*60}")
        print(f"LAYER: {name}")
        print(f"SHAPE: {list(param.shape)}")
        print(f"DTYPE: {param.dtype}")
        print(f"DEVICE: {param.device}")
        print(f"REQUIRES_GRAD: {param.requires_grad}")
        print(f"TOTAL PARAMETERS: {param.numel():,}")
        print(f"{'-'*60}")
        
        # Get parameter values
        param_data = param.data.detach().cpu()
        flattened = param_data.flatten()
        
        # Print statistics
        print(f"MIN VALUE: {flattened.min().item():.6f}")
        print(f"MAX VALUE: {flattened.max().item():.6f}")
        print(f"MEAN VALUE: {flattened.mean().item():.6f}")
        print(f"STD VALUE: {flattened.std().item():.6f}")
        
        # Print first few values
        print(f"\nFIRST {min(max_values_per_layer, param.numel())} VALUES:")
        values_to_show = min(max_values_per_layer, param.numel())
        
        for i in range(values_to_show):
            if i % 5 == 0 and i > 0:  # New line every 5 values
                print()
            print(f"{flattened[i].item():>12.6f}", end=" ")
        
        print()  # New line after values
        
        # For 2D+ tensors, show original shape structure for first few elements
        if len(param.shape) >= 2:
            print(f"\nORIGINAL TENSOR STRUCTURE (first 3x3 if applicable):")
            if len(param.shape) == 2:  # Matrix
                rows_to_show = min(3, param.shape[0])
                cols_to_show = min(3, param.shape[1])
                for i in range(rows_to_show):
                    for j in range(cols_to_show):
                        print(f"{param_data[i, j].item():>10.4f}", end=" ")
                    if param.shape[1] > 3:
                        print("...", end="")
                    print()
                if param.shape[0] > 3:
                    print("...")
                    
            elif len(param.shape) == 4:  # Conv weights (out_channels, in_channels, height, width)
                print(f"Conv weight shape: {param.shape}")
                # Show first channel of first filter
                if param.shape[2] >= 3 and param.shape[3] >= 3:
                    print("First 3x3 of first filter, first input channel:")
                    for i in range(min(3, param.shape[2])):
                        for j in range(min(3, param.shape[3])):
                            print(f"{param_data[0, 0, i, j].item():>10.4f}", end=" ")
                        print()
        
        layer_count += 1
    
    print(f"\n{'='*80}")
    
# Executes the main method from the main.py Python script
if __name__ == '__main__':
    # Calls the main function for the DINOv3 feature extractor (FE) script
    main()
