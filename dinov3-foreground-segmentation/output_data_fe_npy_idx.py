import torch
import numpy as np

class OutputDataFENpyIdx():
    """Python class object to output the X (Features), y (labels), and image index
       from the DINOv3 feature extractor as NumPy .npy files.
       ARGS:
            x_tensor (tensor): tensor of the features X from the DINOv3 feature extractor.
            y_tensor (tensor): tensor of the labels y.
            image_index_tensor (tensor): tensor of the image indices.
            x_tensor_path (str): path to save the X features .npy file.
            y_tensor_path (str): path to save the y labels .npy file.
            image_index_tensor_path (str): path to save the image index .npy file.
    """
    def __init__(self, x_tensor, y_tensor, image_index_tensor,
                 x_tensor_path, y_tensor_path, image_index_tensor_path):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.image_index_tensor = image_index_tensor
        self.x_tensor_path = x_tensor_path
        self.y_tensor_path = y_tensor_path
        self.image_index_tensor_path = image_index_tensor_path

    def tensor_to_npy_features(self):

        x_np = self.x_tensor.cpu().numpy()

        print("Converted NumPy Array Shape")
        print(x_np.shape)

        np.save(self.x_tensor_path, x_np)
        print('Saved features to file.')

    def tensor_to_npy_labels(self):

        y_np = self.y_tensor.cpu().numpy()

        print("Converted NumPy Array Shape")
        print(y_np.shape)

        np.save(self.y_tensor_path, y_np)
        print('Saved labels to file.')

    def tensor_to_npy_image_index(self):

        image_index_np = self.image_index_tensor.cpu().numpy()

        print("Converted NumPy Array Shape")
        print(image_index_np.shape)

        np.save(self.image_index_tensor_path, image_index_np)
        print('Saved image index to file.')
