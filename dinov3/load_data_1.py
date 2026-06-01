import os
from PIL import Image
import numpy as np

class LoadData():

  """
  Load custom image data from the Borebreen glacier in Svalbard Norway 
     to input to the feature extractor of DINOv3.
     ARGS: 
     image_dir (str): Input file path directory of the input image.
     labels_dir (str): Input file path directory of the input label.
     
     RETURNS:
     images (list): Python list of all the input images for DINOv3's feature extractor.
     labels (list): Python list of all the input labels for DINOv3's feature extractor.  
  """

  def __init__(self, image_dir, labels_dir):
    self.image_dir = image_dir
    self.labels_dir = labels_dir

  def load_images(self):

    image = []
    
    # Check if it's an image by extension
    img = Image.open(self.image_dir).convert("RGB")  # ensure RGB format
    image.append(img)

    print(f"Loaded {len(image)} images.")
    print(f"Image 1 Shape {image[0].size}")

    return image

  def load_labels(self):

    # List to store resized images
    label = []
                
    img = Image.open(self.labels_dir).convert('L')
    img = img.resize((1024, 1024), Image.LANCZOS)  # resize to 1024x1024
    label.append(img)

    print(f"Loaded {len(label)} images resized to 1024x1024 (kept original mode).")
    print(f"Labels Shape {label[0].size}")
    
    return label

  def print_image_info(self, label):

    min_val, max_val = label[0].getextrema()
    print(f"Image: min={min_val}, max={max_val}, mode={label[0].mode}, size={label[0].size}")

  def binary_labels_convert(self, label, new_min=0, new_max=255):

    binary_label = []

    if label[0].mode != "L":
      raise ValueError("Image must be in grayscale ('L') mode.")
        
    # Get current min/max
    extrema = label[0].getextrema()
    old_min, old_max = extrema

    print("Label old mininmum pixel value:", old_min)
    print("Label old maximum pixel value:", old_max)

    if old_min == old_max:
      # Avoid divide-by-zero; image is flat
      return label[0].point(lambda p: new_min)

    # Linear rescale
    scale = (new_max - new_min) / (old_max - old_min)
    offset = new_min - old_min * scale

    new_label = label[0].point(lambda p: int(p * scale + offset))

    binary_label.append(new_label)

    new_min, new_max = new_label.getextrema()
    print("Label new mininmum pixel value:", new_min)
    print("Label new maximum pixel value:", new_max)
      
    print(f"Loaded {len(binary_label)} images resized to 1024x1024 (kept original mode).")

    return binary_label

  def sequence_data_loading(self):

    images = self.load_images()

    labels = self.load_labels()

    self.print_image_info(labels)
  
    binary_labels = self.binary_labels_convert(labels)

    return images, binary_labels
