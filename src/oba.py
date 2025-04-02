import cv2
import numpy as np

def extract_object(source_img, polygon, padding=5):
    """
    Given a 12-channel source image and a polygon (list of [x, y] coordinates),
    create a binary mask from the polygon, extract the bounding box, and return
    the cropped object image and its mask.
    """
    h, w, _ = source_img.shape
    # Create an empty mask
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(obj_mask, [pts], color=1)
    
    # Find bounding box coordinates of the object
    ys, xs = np.where(obj_mask == 1)
    if ys.size == 0 or xs.size == 0:
        return None, None  # No object found

    y_min, y_max = max(ys.min() - padding, 0), min(ys.max() + padding, h)
    x_min, x_max = max(xs.min() - padding, 0), min(xs.max() + padding, w)
    
    # Crop the object region from the image and mask
    obj_img = source_img[y_min:y_max, x_min:x_max, :]
    obj_mask_cropped = obj_mask[y_min:y_max, x_min:x_max]
    return obj_img, obj_mask_cropped

def paste_object(target_img, target_mask, obj_img, obj_mask, class_channel):
    """
    Paste the object onto the target image at a random location and update the target mask.
    target_img: (H, W, 12) image (channels-last).
    target_mask: (H, W, 4) segmentation mask (channels-last).
    class_channel: index corresponding to the object's class.
    """
    h_obj, w_obj, _ = obj_img.shape
    H, W, _ = target_img.shape
    if H - h_obj <= 0 or W - w_obj <= 0:
        return target_img, target_mask  # Object too large to paste

    top = np.random.randint(0, H - h_obj)
    left = np.random.randint(0, W - w_obj)
    
    new_img = target_img.copy()
    new_mask = target_mask.copy()
    
    # Expand obj_mask to match image channels for blending
    obj_mask_expanded = np.expand_dims(obj_mask, axis=-1)
    
    # Blend the object: where obj_mask==1, target is replaced by object pixels
    new_img[top:top+h_obj, left:left+w_obj, :] = (
        new_img[top:top+h_obj, left:left+w_obj, :] * (1 - obj_mask_expanded) +
        obj_img * obj_mask_expanded
    )
    
    # Update the corresponding channel in the target mask (channels-last)
    new_mask[top:top+h_obj, left:left+w_obj, class_channel] = np.maximum(
        new_mask[top:top+h_obj, left:left+w_obj, class_channel],
        obj_mask
    )
    
    return new_img, new_mask

