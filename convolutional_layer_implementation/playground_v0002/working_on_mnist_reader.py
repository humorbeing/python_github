import load_mnist as mn
import numpy as np
from PIL import Image

# mn.init()
x_train, t_train, image, label = mn.load()
# print(x_test.shape)
# print(t_test.shape)


print(label[0])


def find_image_with_label(target_label, how_many):
    indexs = []
    num_found = 0
    for i in range(10000):
        if label[i] == target_label:
            indexs.append(i)
            num_found += 1
        if num_found == how_many:
            break
    return indexs
simple_trainset = []
idx = find_image_with_label(7, 10)

for i in range(10):
    im = image[idx[1]]
    im = np.reshape(im, (28, 28))
    simple_trainset.append([im, 7])
simple_trainset = np.array(simple_trainset)
print(simple_trainset.shape)
print(simple_trainset[2][0].shape)
print(simple_trainset[2][1])


def make_simple_trainset(how_many_each, is_save=True, save_path='./saved'):
    simple_trainset = []
    for label in range(10):
        idx = find_image_with_label(label, how_many_each)
        for i in range(how_many_each):
            im = image[idx[i]]
            im = np.reshape(im, (28, 28))
            simple_trainset.append([im, label])
    simple_trainset = np.array(simple_trainset)
    np.random.shuffle(simple_trainset)
    if is_save:
        np.save(save_path, simple_trainset)
    return simple_trainset

make_simple_trainset(1)

ts = np.load('./saved.npy')

print(ts.shape)
for i in ts:
    print(i[1])


def show_im(ss, which):
    im = ss[which][0]
    ll = ss[which][1]
    print('label is',ll)
    new_im = Image.fromarray(im)
    new_im = new_im.resize((280, 280))
    new_im.show()
show_im(ts, 6)

