import torch
import pandas as pd

class OutputDataFE():
    """Python class object to output the X (Features) from the DINOv3 feature extractor
       and y (labels).
       ARGS:
            x_tensor (tensor): tensor of the features X from the DINOv3 feature extractor.
            y_tensor (tensor): tensor of the labels y.
            x_tensor_path (str): path to save the X features CSV file.
            y_tensor_path (str): path to save the y labels CSV file.
    """
    def __init__(self, x_tensor, y_tensor, x_tensor_path, y_tensor_path):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.x_tensor_path = x_tensor_path
        self.y_tensor_path = y_tensor_path

    def tensor_to_df_features(self):

        # Convert to NumPy, then DataFrame
        x_df = pd.DataFrame(self.x_tensor.cpu().numpy())

        print("Converted DataFrame Shape")
        print(x_df.shape)

        x_df.to_csv(self.x_tensor_path, index=False)
        print('Saved features to file.')

    def tensor_to_df_labels(self):

        # Convert to NumPy, then DataFrame
        y_df = pd.DataFrame(self.y_tensor.cpu().numpy(), columns=["labels"])

        print("Converted DataFrame Shape")
        print(y_df.shape)

        y_df.to_csv(self.y_tensor_path, index=False)
        print('Saved labels to file.')
