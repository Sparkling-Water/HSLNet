r""" Helper functions """
import random
import time
import math
import cv2
import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


# ========================================
# OTSU
# From https://github.com/vision-kek/abcdfss
# ========================================
norm = lambda t: (t - t.min()) / (t.max() - t.min())
denorm = lambda t, min_, max_: t * (max_ - min_) + min_

def otsus(batched_tensor_image, drop_least=0.05):
    bsz = batched_tensor_image.size(0)
    binary_tensors = []
    thresholds = []

    for i in range(bsz):
        # Convert the tensor to numpy array
        numpy_image = batched_tensor_image[i].cpu().numpy()

        # Rescale to [0, 255] and convert to uint8 type for OpenCV compatibility
        npmin, npmax = numpy_image.min(), numpy_image.max()
        numpy_image = (norm(numpy_image) * 255).astype(np.uint8)

        # Drop values that are in the lowest percentiles
        truncated_vals = numpy_image[numpy_image >= int(255 * drop_least)]

        # Apply Otsu's thresholding
        thresh_value, _ = cv2.threshold(truncated_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply the computed threshold on the original image
        binary_image = (numpy_image > thresh_value).astype(np.uint8) * 255

        # Convert the result back to a tensor and append to the list
        binary_tensors.append(torch.from_numpy(binary_image).float() / 255)

        thresholds.append(torch.tensor(denorm(thresh_value / 255, npmin, npmax)) \
                          .to(batched_tensor_image.device, dtype=batched_tensor_image.dtype))

    # Convert list of tensors back to a single batched tensor
    binary_tensor_batch = torch.stack(binary_tensors, dim=0)
    thresh_batch = torch.stack(thresholds, dim=0)
    return thresh_batch, binary_tensor_batch


def calcthresh(fused_pred, method='otsus'):
    if method == 'otsus':
        thresh = otsus(fused_pred)[0]
        return thresh
    elif method == 'pred_mean':
        otsu_thresh = otsus(fused_pred)[0]
        thresh = torch.max(otsu_thresh, fused_pred.mean())
    return thresh
