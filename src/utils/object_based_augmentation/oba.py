import cv2
import numpy as np

def extract_object(source_img, polygon, padding=5):
    """
    Given a 12-channel source image and a polygon (list of [x, y] coordinates),
    create a binary mask from the polygon, extract the bounding box, and return
    the cropped object image and its mask.

    Parameters:
        source_img      : np.ndarray of shape (H, W, C), the full image from which to extract.
        polygon         : list of floats, flat list like [x0, y0, x1, y1, ..., xn, yn] defining a closed polygon.
        padding         : int, default=5, number of pixels to pad the bounding box on each side.
    
    Returns:
        obj_img : np.ndarray of shape (h, w, C), or None if polygon empty
            The cropped image region containing the object.
        obj_mask: np.ndarray of shape (h, w), or None if polygon empty
            Binary mask (0/1) of the object within the crop.
    """
    # Get source image shape
    h, w, _ = source_img.shape

    # Create an empty mask
    obj_mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(obj_mask, [pts], color=1)
    
    # Find bounding box coordinates of the object
    ys, xs = np.where(obj_mask == 1)
    if ys.size == 0 or xs.size == 0:
        return None, None   # No object found

    y_min, y_max = max(ys.min() - padding, 0), min(ys.max() + padding, h)
    x_min, x_max = max(xs.min() - padding, 0), min(xs.max() + padding, w)
    
    # Extract the object region from the image and mask
    obj_img = source_img[y_min:y_max, x_min:x_max, :]
    obj_mask_cropped = obj_mask[y_min:y_max, x_min:x_max]

    return obj_img, obj_mask_cropped

def paste_object(target_img, target_mask, obj_img, obj_mask, class_channel, max_attempts=10, highlight=False):
    """
    Paste the object onto the target image at a random location where no segmentation objects 
    are present in the target mask, and update the target mask.

    Parameters:
        target_img      : numpy array of shape (H, W, C), the base image.
        target_mask     : numpy array of shape (H, W, K), the base segmentation mask.
        obj_img         : numpy array of shape (h, w, C), the cut-out object image.
        obj_mask        : numpy array of shape (h, w), binary mask for the object.
        class_channel   : int, which channel in target_mask to update (0 ≤ class_channel < K).
        max_attempts    : int, how many random placements to try before giving up.
        highlight       : bool, if True return the paste bbox for visualization.

    Returns:
        If highlight is True:
            new_img, new_mask, bbox   where bbox = (top, left, height_of_obj, width_of_obj)
        Otherwise:
            new_img, new_mask.
    """
    h_obj, w_obj = obj_img.shape[:2]

    H, W, _ = target_img.shape
    if H - h_obj <= 0 or W - w_obj <= 0:
        if highlight:
            return target_img, target_mask, None
        return target_img, target_mask

    # Create copies and set variables
    new_img = target_img.copy()
    new_mask = target_mask.copy()
    location_found = False
    chosen_top, chosen_left = None, None

    # Try to find a candidate location with no segmentation at all (so we don't create unnatural overlapping segmentations)
    for _attempt in range(max_attempts):
        top = np.random.randint(0, H - h_obj)
        left = np.random.randint(0, W - w_obj)
        
        # Extract the candidate region from the target mask (all channels)
        roi = new_mask[top:top + h_obj, left:left + w_obj, :]
        
        # Check that the entire region is empty (all zeros)
        if np.all(roi == 0):
            chosen_top, chosen_left = top, left
            location_found = True
            break

    # If no conflict-free location is found, skip the pasting of object
    if not location_found:
        if highlight:
            return target_img, target_mask, None
        return target_img, target_mask

    # Paste the object and update mask in the chosen candidate region
    obj_mask_expanded = np.expand_dims(obj_mask, axis =- 1)
    new_img[chosen_top:chosen_top + h_obj, chosen_left:chosen_left + w_obj, :] = (
        new_img[chosen_top:chosen_top + h_obj, chosen_left:chosen_left + w_obj, :] * (1 - obj_mask_expanded) +
        obj_img * obj_mask_expanded
    )

    # Update only the appropriate segmentation channel
    new_mask[chosen_top:chosen_top + h_obj, chosen_left:chosen_left + w_obj, class_channel] = np.maximum(
        new_mask[chosen_top:chosen_top + h_obj, chosen_left:chosen_left + w_obj, class_channel],
        obj_mask
    )
    
    # If function is used for visualization purposes, create a highlighting boundingbox on top of the image
    if highlight:
        bbox = (chosen_top, chosen_left, h_obj, w_obj)
        return new_img, new_mask, bbox
    
    return new_img, new_mask
