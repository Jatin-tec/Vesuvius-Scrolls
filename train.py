import time
from . import get_device, get_gpu_memory, clear_gpu_memory, dice_score
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms


import os
import torch


def train_loop(
    train_dir: str = "data/train/",
    weights_filepath: str = None,
    optimizer: str = "adam",
    curriculum: str = "1",
    num_samples: int = 100,
    output_dir: str = "output/train",
    image_augs: bool = False,
    input_channels: int = 1,
    output_channels: int = 3,
    patch_size_x: int = 512,
    patch_size_y: int = 128,
    resize_ratio: float = 1.0,
    batch_size: int = 16,
    lr: float = 0.001,
    lr_gamma: float = None,
    num_epochs: int = 2,
    num_workers: int = 0,
    writer: bool = False,
    save_model: bool = False,
    max_time_hours: float = 8,
):
    # Notebook will only run for this amount of time
    print(f"Training will run for {max_time_hours} hours")
    time_train_max_seconds = max_time_hours * 60 * 60
    time_start = time.time()
    time_elapsed = 0

    # Get GPU
    get_gpu_memory()
    device = get_device()
    clear_gpu_memory()

    # Load the model, try to fit on GPU
    model = AttentionUNet(
        in_channel=input_channels,
    )

    if weights_filepath is not None:
        print(f"Loading weights from {weights_filepath}")
        model.load_state_dict(torch.load(weights_filepath))

    model = model.to(device)
    model.train()

    # Create optimizers
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss()

    if lr_gamma is not None:
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_gamma)

    # Writer for Tensorboard
    if writer:
        writer = SummaryWriter(output_dir)

    # Train the model
    best_score = 0
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")

        # Curriculum defines the order of the training
        for current_dataset_id in curriculum:

            _train_dir = os.path.join(train_dir, current_dataset_id)
            print(f"Training on dataset: {_train_dir}")

            # Training dataset
            train_dataset = FragmentDataset(
                # Directory containing the dataset
                _train_dir,
                # Expected slices per fragment
                slice_depth=input_channels,
                # Size of an individual patch
                patch_size_x=patch_size_x,
                patch_size_y=patch_size_y,
                # Image resize ratio
                resize_ratio=resize_ratio,
                # Training vs Testing mode
                train=True,
            )

            total_dataset_size = len(train_dataset)
            print(f"Raw train dataset size: {total_dataset_size}")

            # Add augmentations
            img_transform_list = [
                transforms.Normalize(train_dataset.mean, train_dataset.std)
            ]

            if image_augs:
                img_transform_list += [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]

            img_transform = transforms.Compose(img_transform_list)

            # DataLoaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=torch.utils.data.default_collate,
                sampler=RandomSampler(
                    train_dataset, num_samples=num_samples),
                num_workers=num_workers,
                # This will make it go faster if it is loaded into a GPU
                pin_memory=True,
            )

            print(f"Training...")
            train_loss = 0
            score = 0
            for patch, label in train_dataloader:
                optimizer.zero_grad()
                # writer.add_histogram('patch_input', patch, step)
                # writer.add_histogram('label_input', label, step)
                patch = patch.to(device)
                patch = img_transform(patch)
                pred = model(patch)
                label = label.to(device)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                step += 1
                train_loss += loss.item()
                score += dice_score(pred, label)

                # Check if we have exceeded the time limit
                time_elapsed = time.time() - time_start
                print(f"Time elapsed: {time_elapsed} seconds")
                if time_elapsed > time_train_max_seconds:
                    print("Time limit exceeded, stopping batches")
                    break

            if writer:
                train_loss /= num_samples
                writer.add_scalar(
                    f'{loss_fn.__class__.__name__}/{current_dataset_id}/train', train_loss, step)

            # Score is average dice score for all batches
            score /= len(train_dataloader)
            if score > best_score:
                print("New best score: %.4f" % score)
                best_score = score
                if save_model:
                    print("Saving model...")
                    torch.save(model.state_dict(), f"{output_dir}/model.pth")
            if writer:
                writer.add_scalar(
                    f'Dice/{current_dataset_id}/train', score, step)

            # Check if we have exceeded the time limit
            time_elapsed = time.time() - time_start
            if time_elapsed > time_train_max_seconds:
                print("Time limit exceeded, stopping curriculum")
                break

        if lr_gamma is not None:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" %
                  (epoch, before_lr, after_lr))

        # Check if we have exceeded the time limit
        time_elapsed = time.time() - time_start
        if time_elapsed > time_train_max_seconds:
            print("Time limit exceeded, stopping training")
            break

    if writer:
        writer.close()  # Close the SummaryWriter

    return best_score, model, writer
