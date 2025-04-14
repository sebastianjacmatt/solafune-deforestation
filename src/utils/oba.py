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

def paste_object(target_img, target_mask, obj_img, obj_mask, class_channel, max_attempts=10, highlight=False):
    """
    Paste the object onto the target image at a random location where no segmentation objects 
    are present in the target mask, and update the target mask.

    Parameters:
        target_img: (H, W, 12) numpy array (channels-last).
        target_mask: (H, W, 4) numpy array (channels-last) with segmentation information.
        obj_img: The object image to paste (e.g. extracted from a polygon).
        obj_mask: The binary mask corresponding to obj_img.
        class_channel: The index for the object’s segmentation channel.
        max_attempts: Maximum number of attempts to find a conflict-free location.
        highlight: If True, returns the bounding box of the pasted object.

    Returns:
        If highlight is True:
            new_img, new_mask, bbox   where bbox = (top, left, height_of_obj, width_of_obj)
        Otherwise:
            new_img, new_mask.
    """
    # ----- Apply random rotation (without scaling) to the object -----
    angle = np.random.uniform(0, 360)
    h_obj, w_obj, _ = obj_img.shape
    center = (w_obj // 2, h_obj // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h_obj * sin) + (w_obj * cos))
    nH = int((h_obj * cos) + (w_obj * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated_obj_img = cv2.warpAffine(obj_img, M, (nW, nH), flags=cv2.INTER_LINEAR)
    rotated_obj_mask = cv2.warpAffine(obj_mask, M, (nW, nH), flags=cv2.INTER_NEAREST)
    obj_img = rotated_obj_img
    obj_mask = rotated_obj_mask
    h_obj, w_obj, _ = obj_img.shape
    # ----- End rotation step -----

    H, W, _ = target_img.shape
    if H - h_obj <= 0 or W - w_obj <= 0:
        if highlight:
            return target_img, target_mask, None
        return target_img, target_mask

    new_img = target_img.copy()
    new_mask = target_mask.copy()
    location_found = False
    chosen_top, chosen_left = None, None

    # Try to find a candidate location with no segmentation at all.
    for attempt in range(max_attempts):
        top = np.random.randint(0, H - h_obj)
        left = np.random.randint(0, W - w_obj)
        
        # Extract the candidate region from the target mask (all channels)
        roi = new_mask[top:top+h_obj, left:left+w_obj, :]
        
        # Check that the entire region is empty (all zeros)
        if np.all(roi == 0):
            chosen_top, chosen_left = top, left
            location_found = True
            break

    # If no conflict-free location is found, skip pasting.
    if not location_found:
        if highlight:
            return target_img, target_mask, None
        return target_img, target_mask

    # Paste object and update mask in the chosen candidate region.
    obj_mask_expanded = np.expand_dims(obj_mask, axis=-1)
    new_img[chosen_top:chosen_top+h_obj, chosen_left:chosen_left+w_obj, :] = (
        new_img[chosen_top:chosen_top+h_obj, chosen_left:chosen_left+w_obj, :] * (1 - obj_mask_expanded) +
        obj_img * obj_mask_expanded
    )
    # Update only the appropriate segmentation channel.
    new_mask[chosen_top:chosen_top+h_obj, chosen_left:chosen_left+w_obj, class_channel] = np.maximum(
        new_mask[chosen_top:chosen_top+h_obj, chosen_left:chosen_left+w_obj, class_channel],
        obj_mask
    )
    
    if highlight:
        bbox = (chosen_top, chosen_left, h_obj, w_obj)
        return new_img, new_mask, bbox
    return new_img, new_mask
