
from PIL import Image
from torchvision import transforms



def open_image_ok_size(image_path, size=512):
    image = Image.open(image_path)
    image = transforms.Resize(size)(image)
    w, h = image.size
    image = transforms.CenterCrop(((h // 16) * 16, (w // 16) * 16))(image)
    return image