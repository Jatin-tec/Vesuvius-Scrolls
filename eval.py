# imports
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from . import get_device, get_gpu_memory, clear_gpu_memory, dice_score
from . import CNNModel
from . import FragmentDataset


def eval(
    eval_dir: str = "data/eval/",
    model: str = "CNN",
    output_dir: str = "output/train",
    weights_filepath: str = None,
    slice_depth: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    batch_size: int = 16,
    num_workers: int = 1,
    save_pred_img: bool = False,
    save_submit_csv: bool = False,
    threshold: float = 0.5,
    postprocess: bool = True,
    writer: SummaryWriter = None,
):
    # Get GPU
    device = get_device()
    clear_gpu_memory()

    # Load the model, try to fit on GPU
    if isinstance(model, str):
        model = AttentionUNet(
            in_channel=slice_depth,
        )
        if weights_filepath is not None:
            print(f"Loading weights from {weights_filepath}")
            model.load_state_dict(torch.load(weights_filepath))

    model = model.to(device)
    model.eval()

    if save_submit_csv:
        submission_filepath = os.path.join(output_dir, 'submission.csv')
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted\n")

    # Baseline is to use image mask to create guess submission
    for subtest_name in ['a', 'b']:

        # Name of sub-directory inside test dir
        subtest_filepath = os.path.join(eval_dir, subtest_name)

        # Evaluation dataset
        eval_dataset = FragmentDataset(
            # Directory containing the dataset
            subtest_filepath,
            # Expected slices per fragment
            slice_depth=slice_depth,
            # Size of an individual patch
            patch_size_x=patch_size_x,
            patch_size_y=patch_size_y,
            # Image resize ratio
            resize_ratio=resize_ratio,
            # Training vs Testing mode
            train=False,
        )

        # DataLoaders
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(eval_dataset),
            num_workers=num_workers,
            # This will make it go faster if it is loaded into a GPU
            pin_memory=True,
        )

        img_transform = transforms.Compose([
            transforms.Normalize(eval_dataset.mean, eval_dataset.std)
        ])

        # Make a blank prediction image
        pred_image = np.zeros(eval_dataset.resized_size +
                              (1,), dtype=np.float32).T
        # Convert the NumPy array to a PyTorch tensor
        pred_image = torch.from_numpy(pred_image)

        patch_size_y_half = patch_size_y // 2
        patch_size_x_half = patch_size_x // 2

        # Pad Prediction
        pred_image = F.pad(
            pred_image,
            (
                # Padding in Y
                patch_size_y_half, patch_size_y_half,
                # Padding in X
                patch_size_x_half, patch_size_x_half,
                # no padding in Y
                0, 0
            ),
            mode='constant',
            value=0,).to(device)

        print(f"Prediction image {subtest_name} shape: {pred_image.shape}")
        print(
            f"Prediction image min: {pred_image.min()}, max: {pred_image.max()}")

        for i, batch in enumerate(tqdm(eval_dataloader)):
            batch = batch.to(device)
            batch = img_transform(batch)
            with torch.no_grad():
                preds = model(batch)

            for j, pred in enumerate(preds):
                pixel_index = eval_dataset.mask_indices[i * batch_size + j]
                x, y = pixel_index
                pred_image[:, x:x+patch_size_x, y:y+patch_size_x] = pred

        # Calculate the indices for cropping the tensor
        start_x = patch_size_x_half
        end_x = pred_image.size()[-2] - patch_size_x_half
        start_y = patch_size_y_half
        end_y = pred_image.size()[-1] - patch_size_y_half

        # remove padding from prediction image
        pred_image = pred_image[:, start_x:end_x, start_y:end_y]

        # Convert the PyTorch tensor to a NumPy array
        pred_image = pred_image.squeeze().cpu().numpy()

        # 'gray' colormap for grayscale images
        plt.imshow(pred_image, cmap='gray')
        plt.axis('off')  # Turn off axis ticks
        plt.show()

        img = Image.fromarray(pred_image * 255).convert('1')
        img = img.resize((
            eval_dataset.original_size[0],
            eval_dataset.original_size[1],
        ), resample=Image.BILINEAR)

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}.png")
            img.save(_image_filepath)

        if postprocess:
            print("Postprocessing...")
            # Erosion then Dilation
            _filter_size = 3
            img = img.filter(ImageFilter.MinFilter(_filter_size))
            img = img.filter(ImageFilter.MaxFilter(_filter_size))

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(
                output_dir, f"pred_{subtest_name}_post.png")
            img.save(_image_filepath)

        if save_submit_csv:
            print("Saving submission csv...")
            starts_ix, lengths = image_to_rle(
                np.array(img), threshold=threshold)
            inklabels_rle = " ".join(
                map(str, sum(zip(starts_ix, lengths), ())))
            print("Writing")
            with open(submission_filepath, 'a') as f:
                f.write(f"{subtest_name},{inklabels_rle}\n")

    print("Done")
    if save_pred_img and save_submit_csv:
        save_rle_as_image(submission_filepath, output_dir,
                          subtest_name, pred_image.shape)
