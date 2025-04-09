from albumentations import (
    HorizontalFlip, ShiftScaleRotate, RandomRotate90, OneOf, Compose
)
from albumentations.augmentations.transforms import (
    CLAHE,  RandomGamma,
     Sharpen, #Blur, 
    RandomBrightnessContrast, GaussNoise,
    #OpticalDistortion,
    #MotionBlur,
    #MedianBlur,
)


def augmentation(img, mask, color_aug_prob):
    transforms = [
        GaussNoise(p=1),
        #OpticalDistortion(p=1), 
        RandomBrightnessContrast(p=1),
        RandomGamma(p=1),
        #Blur(p=1),
        Sharpen(p=1),  # <== replaces IAAEmboss
    ]

    if img.shape[-1] in [1, 3]:
        transforms.insert(0, CLAHE(p=1))  # Safe check for 3-channel images

    aug_func = Compose([
        RandomRotate90(p=0.8),
        HorizontalFlip(p=0.6),
        OneOf(transforms, p=color_aug_prob)
    ], p=0.9)

    mask = mask.astype('uint8')
    data = {"image": img, "mask": mask}
    augmented = aug_func(**data)
    image, mask = augmented["image"], augmented["mask"]
    return image, mask

