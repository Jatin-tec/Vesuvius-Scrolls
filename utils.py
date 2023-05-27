import torch 
import csv
import subprocess
import numpy as np
import gc
import os
from PIL import Image

def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def image_to_rle(img, threshold=0.5):
    # Convert image to binary mask
    binary_mask = (img > threshold).astype(np.uint8)

    # Find the starting and ending indices of runs
    starts = (binary_mask[:-1] == 0) & (binary_mask[1:] == 1)
    ends = (binary_mask[:-1] == 1) & (binary_mask[1:] == 0)

    # Calculate the starting and ending indices
    starts_ix = np.where(starts)[0] + 2
    lengths = np.where(ends)[0] - starts_ix + 1

    return starts_ix, lengths

def save_rle_as_image(rle_csv_path, output_dir, subtest_name, image_shape):
    with open(rle_csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            _subtest_name, rle_data = row
            if _subtest_name != subtest_name:
                continue
            rle_pairs = list(map(int, rle_data.split()))

            # Decode RLE data
            img = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
            for i in range(0, len(rle_pairs), 2):
                start = rle_pairs[i] - 1
                end = start + rle_pairs[i + 1]
                img[start:end] = 1

            # Reshape decoded image data to original shape
            img = img.reshape(image_shape)
            img = Image.fromarray(img * 255).convert('1')
            _image_filepath = os.path.join(output_dir, f"pred_{subtest_name}_rle.png")
            img.save(_image_filepath)


def dice_score(preds, label, beta=0.5, epsilon=1e-6):
    # Implementation of DICE coefficient
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    print(f"Predictions tensor shape: {preds.shape}")
    print(f"Predictions tensor dtype: {preds.dtype}")
    print(f"Predictions tensor min: {preds.min()}")
    print(f"Predictions tensor max: {preds.max()}")
    label = label.flatten()
    print(f"Label tensor shape: {label.shape}")
    print(f"Label tensor dtype: {label.dtype}")
    print(f"Label tensor min: {label.min()}")
    print(f"Label tensor max: {label.max()}")
    tp = preds[label==1].sum()
    fp = preds[label==0].sum()
    fn = label.sum() - tp
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    _score = (1 + beta * beta) * (p * r) / (beta * beta * p + r + epsilon)
    print(f"DICE score: {_score}")
    return _score


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                             '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
    gpu_memory = [tuple(map(int, line.split(',')))
                  for line in result.stdout.strip().split('\n')]
    for i, (used, free) in enumerate(gpu_memory):
        print(f"GPU {i}: Memory Used: {used} MiB | Memory Available: {free} MiB")


def clear_gpu_memory():
    if torch.cuda.is_available():
        print('Clearing GPU memory')
        torch.cuda.empty_cache()
        gc.collect()