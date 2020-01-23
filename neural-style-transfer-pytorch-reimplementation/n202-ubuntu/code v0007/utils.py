from torchvision import transforms
from PIL import Image

def ss(s): raise Exception(s)

size = 1000
def open_image_ok_size(image_path, size=size):
    image = Image.open(image_path)
    image = transforms.Resize(size)(image)
    w, h = image.size
    image = transforms.CenterCrop(((h // 16) * 16, (w // 16) * 16))(image)
    return image

def load_image(filename, size=size):
    img = open_image_ok_size(filename, size=size)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    # print(y.shape)
    # print(y.size())

    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)

    features_t = features.transpose(1, 2)
    # print(features.shape)
    # print(features_t.shape)
    gram = features.bmm(features_t) / (ch * h * w)
    # print(ch*h*w)
    # ss('in gram matrix')
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    # print(batch.shape)
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    # print(mean.shape)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
