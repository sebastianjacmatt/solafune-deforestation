import cv2
import numpy as np

def rotate_object(obj_img, obj_mask):
    """Randomly rotate object without scaling."""
    angle = np.random.uniform(0, 360)
    h_obj, w_obj = obj_img.shape[:2]
    center = (w_obj // 2, h_obj // 2)

    # build rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW = int(h_obj * sin + w_obj * cos)
    nH = int(h_obj * cos + w_obj * sin)
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # warp image and mask
    rotated_img  = cv2.warpAffine(obj_img,  M, (nW, nH), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(obj_mask, M, (nW, nH), flags=cv2.INTER_NEAREST)

    return rotated_img, rotated_mask


def flip_object(obj_img, obj_mask):
    """Randomly flip horizontally, vertically, or both."""
    # flipCode:  0 = X-axis; 1 = Y-axis; -1 = both
    choice = np.random.choice([0, 1, -1, None], p=[0.3, 0.3, 0.3, 0.1])
    if choice is None:
        return obj_img, obj_mask

    flipped_img  = cv2.flip(obj_img,  choice)
    flipped_mask = cv2.flip(obj_mask, choice)
    return flipped_img, flipped_mask


def blend_in_object(obj_img, obj_mask, ksize=15, feather_radius=50):
    """
    Create a smooth fade between original and blurred using distance transforms.
    - ksize: gaussian blur kernel size.
    - feather_radius: how many pixels the blur fades over.
    """
    # 1) full-object blur
    blurred = cv2.GaussianBlur(obj_img, (ksize, ksize), sigmaX=0)

    # 2) distance to mask edge
    # dt_out: dist of each bg pixel to nearest object
    # dt_in:  dist of each object pixel to nearest bg
    dt_out = cv2.distanceTransform(1 - obj_mask, cv2.DIST_L2, 5)
    dt_in  = cv2.distanceTransform(obj_mask,     cv2.DIST_L2, 5)
    # we only care about a band of width `feather_radius` around the edge:
    weight = np.clip(dt_out / feather_radius, 0, 1) * obj_mask \
           + np.clip(dt_in  / feather_radius, 0, 1) * (1 - obj_mask)

    # smooth the weight map a bit
    weight = cv2.GaussianBlur(weight, (ksize, ksize), sigmaX=0)
    weight3 = weight[:, :, None].repeat(obj_img.shape[2], axis=2)

    # 3) composite with a soft ramp
    out_img = (obj_img * (1 - weight3) + blurred * weight3).astype(obj_img.dtype)
    return out_img, obj_mask



def augment_object(obj_img, obj_mask):
    """
    Run all object-level augmentations in sequence.
    """
    img, m = rotate_object(obj_img, obj_mask)
    img, m = flip_object(img, m)
    img, m = blend_in_object(img, m)
    return img, m
