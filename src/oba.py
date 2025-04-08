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

def paste_object(target_img, target_mask, obj_img, obj_mask, class_channel, highlight=False):
    """
    Paste the object onto the target image at a random location and update the target mask. target_img: (H, W, 12) image (channels-last).
    target_mask: (H, W, 4) segmentation mask (channels-last). class_channel: index corresponding to the object's class.
    This updated function randomly rotates the extracted object (and its mask) before pasting.
    """
    # Choose a random rotation angle in degrees
    angle = np.random.uniform(0, 360)

    # Get dimensions and center of the object image
    h_obj, w_obj, _ = obj_img.shape
    center = (w_obj // 2, h_obj // 2)

    # Compute rotation matrix (with no scaling, scale=1.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Compute the new bounding dimensions of the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h_obj * sin) + (w_obj * cos))
    nH = int((h_obj * cos) + (w_obj * sin))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Rotate the object image using linear interpolation
    rotated_obj_img = cv2.warpAffine(obj_img, M, (nW, nH), flags=cv2.INTER_LINEAR)
    # Rotate the mask using nearest neighbor to preserve binary values
    rotated_obj_mask = cv2.warpAffine(obj_mask, M, (nW, nH), flags=cv2.INTER_NEAREST)
    # Use the rotated versions for further processing
    obj_img = rotated_obj_img
    obj_mask = rotated_obj_mask
    h_obj, w_obj, _ = obj_img.shape

    H, W, _ = target_img.shape
    if H - h_obj <= 0 or W - w_obj <= 0:
        if highlight:
            return target_img, target_mask, None  # Object too large; no bbox.
        return target_img, target_mask

    top = np.random.randint(0, H - h_obj)
    left = np.random.randint(0, W - w_obj)
    
    new_img = target_img.copy()
    new_mask = target_mask.copy()
    
    # Expand obj_mask to match image channels for blending
    obj_mask_expanded = np.expand_dims(obj_mask, axis=-1)
    new_img[top:top+h_obj, left:left+w_obj, :] = (
        new_img[top:top+h_obj, left:left+w_obj, :] * (1 - obj_mask_expanded) +
        obj_img * obj_mask_expanded
    )
    
    # Update the corresponding channel in the target mask (channels-last)
    new_mask[top:top+h_obj, left:left+w_obj, class_channel] = np.maximum(
        new_mask[top:top+h_obj, left:left+w_obj, class_channel],
        obj_mask
    )
    
    if highlight:
        # Return the bounding box coordinates along with the new image and mask.
        return new_img, new_mask, (top, left, h_obj, w_obj)
    
    return new_img, new_mask
