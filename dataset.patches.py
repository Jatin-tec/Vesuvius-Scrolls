import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class FragmentDataset(Dataset):
    def __init__(
        self,
        # Directory containing the dataset
        data_dir: str,
        
        # Filenames of the images we'll use
        image_mask_filename="mask.png",
        image_labels_filename="inklabels.png",
        slices_dir_filename="surface_volume",
        image_ir_filename="ir.png",
        
        # Expected slices per fragment
        slice_depth: int = 13,

        # Size of an individual patch 
        patch_size_x: int = 64,
        patch_size_y: int = 64,

        #Image resize ratio
        resize_ratio: float = 1.0,

        #Dataset datatype
        dtype: torch.dtype = torch.float32,

        # Training vs Testing mode
        train: bool = True,

        #visualize the patches
        viz: bool = False,

        # Device to use
        device: str = "cuda",
    ):
        print(f"Creating CurriculumDataset for {data_dir}")
        
        self.viz = viz 
        self.device = device
        self.dtype = dtype

        # Train mode also loads the labels
        self.train = train
        
        # Resize ratio reduces the size of the image
        self.resize_ratio = resize_ratio
        
        # Data will be B x slice_depth x patch_size_x x patch_size_y
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

        self.patch_size_x_half = int(patch_size_x / 2)
        self.patch_size_y_half = int(patch_size_y / 2)
        self.slice_depth = slice_depth
        
        assert os.path.exists(
            data_dir), f"Data directory {data_dir} does not exist"
        
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        _mask_img = Image.open(_image_mask_filepath).convert("1")
       
        # Get original size and resized size 
        self.original_size = _mask_img.size
        self.resized_size = (
            int(self.original_size[0] * self.resize_ratio),
            int(self.original_size[1] * self.resize_ratio),
        )
        
        # Resize the mask
        _mask_img = _mask_img.resize(self.resized_size, resample=Image.BILINEAR)
        _mask = torch.from_numpy(np.array(_mask_img)).to(torch.bool)
        self._mask_img = _mask_img
        
        if train:
            _image_labels_filepath = os.path.join(
                data_dir, image_labels_filename)
            
            _labels_img = Image.open(_image_labels_filepath).convert("1")
            _labels_img = _labels_img.resize(
                self.resized_size, resample=Image.BILINEAR)
            self._labels_img=_labels_img
            self.labels = torch.from_numpy(
                np.array(self._labels_img)).to(torch.bool)
            
            self.labels = F.pad(
            self.labels,
            (
                # Padding in Y
                self.patch_size_y_half, self.patch_size_y_half,
                # Padding in X
                self.patch_size_x_half, self.patch_size_x_half,
            ),
            mode='constant',
            value=0,
        )
        
        # Pre-allocate the entire fragment
        self.fragment = torch.zeros((
            self.slice_depth,
            self.resized_size[1],
            self.resized_size[0],
        ), dtype=torch.float32)

        # Open up slices
        _slice_dir = os.path.join(data_dir, slices_dir_filename)
        for i in tqdm(range(self.slice_depth)):
            _slice_filepath = os.path.join(_slice_dir, f"{i:02d}.tif")
            _slice_img = Image.open(_slice_filepath).convert('F')

            # Resize the slice
            _slice_img = _slice_img.resize(
                self.resized_size, resample=Image.BILINEAR)
            
            # Convert to tensor
            _slice = torch.from_numpy(np.array(_slice_img)/65535.0)

            # Store the slice
            self.fragment[i, :, :] = _slice

        # Get mean/std for fragment only on mask indices
        _fragment_mask = _mask.unsqueeze(0).expand(self.slice_depth, -1, -1)
        self.mean = self.fragment[_fragment_mask].mean()
        self.std = self.fragment[_fragment_mask].std()

        # TODO: Use Predictions to additionally balance the dataset
        # if self.train:
        #     # Get indices where labels are 1
        #     self.labels_indices = torch.nonzero(self.labels).to(torch.int32)
        #     # print(f"Labels indices shape: {self.labels_indices.shape}")
        #     # print(f"Labels indices dtype: {self.labels_indices.dtype}")
            
        #     # Indices where mask is 0 and labels is 1
        #     self.mask_0_labels_1_indices = torch.nonzero(
        #         (~_mask) & self.labels
        #     ).to(torch.int32)

        # Pad the fragment with zeros based on patch size
        print(f"Fragment tensor shape before padding: {self.fragment.shape}")
        
        self.fragment = F.pad(
            self.fragment,
            (
                # Padding in Y
                self.patch_size_y_half, self.patch_size_y_half,
                # Padding in X
                self.patch_size_x_half, self.patch_size_x_half,
                # No padding on z
                0, 0,
            ),
            mode='constant',
            value=0,
        )
        
        # Get indices where mask is 1
        self.mask_indices = torch.nonzero(_mask).to(torch.int32)
        
        print(f"Fragment tensor shape: {self.fragment.shape}")
        print(f"Fragment tensor dtype: {self.fragment.dtype}")


    def __len__(self):
        return self.mask_indices.shape[0]

    def __getitem__(self, index):

        # Get the x, y from the mask indices
        x, y = self.mask_indices[index]
        #print(f"Index: {index}, x: {x}, y: {y}")

        # Pre-allocate the patch
        patch = self.fragment[
                :,
                x: x + self.patch_size_x,
                y: y + self.patch_size_y,
        ]
        
        if self.train:
            label = self.labels[
                x: x + self.patch_size_x,
                y: y + self.patch_size_y,
            ].to(torch.float32).view(1, self.patch_size_x, self.patch_size_y)
           
        # Plotting the images and bounding box
        if self.viz:
            
            print(f"Patch tensor shape: {patch.shape}")
            print(f"Patch tensor dtype: {patch.dtype}")
            print(f"Patch tensor min: {patch.min()}")
            print(f"Patch tensor max: {patch.max()}")
            
            
            print(f"Label tensor shape: {label.shape}")
            print(f"Label tensor dtype: {label.dtype}")
            print(f"Label tensor min: {label.min()}")
            print(f"Label tensor max: {label.max()}")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.subplots_adjust(wspace=0.4)

            # Plot resized mask image
            axes[0].imshow(self._mask_img, cmap='gray')
            axes[0].set_title("Resized Mask Image")

            # Plot label image
            axes[1].imshow(self._labels_img, cmap='gray')
            axes[1].set_title("Label Image")

            # Plot slice of index 0 of the fragment
            axes[2].imshow(self.fragment[0], cmap='gray')
            axes[2].set_title("Slice of Index 0")

            # Plot bounding box in each image
            rect = plt.Rectangle((y, x), self.patch_size_y, self.patch_size_x,
                                 edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
            
            rect = plt.Rectangle((y, x), self.patch_size_y, self.patch_size_x,
                                 edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
            
            rect = plt.Rectangle((y, x), self.patch_size_y, self.patch_size_x,
                                 edgecolor='r', facecolor='none')
            axes[2].add_patch(rect)
            
            plt.show()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plt.subplots_adjust(wspace=0.4)
            
            # Plot resized mask image
            axes[0].imshow(patch[0], cmap='gray')
            axes[0].set_title("Patch Fragment")

            # Plot label image
            axes[1].imshow(label.squeeze(), cmap='binary')
            axes[1].set_title("Label Patch")

            plt.show()
            
        if self.train:
            return patch, label
        
        else:
            # If we're not training, we don't have labels
            return patch