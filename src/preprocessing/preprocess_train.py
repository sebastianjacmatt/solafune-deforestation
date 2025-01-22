from PIL import Image
with Image.open("../data/train_images/mushroom.jpg") as im:
    im.show()