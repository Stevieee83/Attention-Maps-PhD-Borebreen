import timm

# List all DINOv3 models
dinov3_models = timm.list_models('*dinov3*')

print("Available DINOv3 models:")
for model in sorted(dinov3_models):
    print(f"  - {model}")

print(f"\nTotal: {len(dinov3_models)} DINOv3 models found")
