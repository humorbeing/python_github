
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

trainset = np.load('./simpleset_1per.npy')
imgs = []
for i in trainset:
    imgs.append(i[0])

imgs = np.array(imgs)
imgs = imgs / 255.

def show_images(imgs):
    show_size = len(imgs)
    plt.gray()
    fig, axs = plt.subplots(1, show_size,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    # axs[0].set_title('Epoch:' + str(epoch))
    for n in range(show_size):
        axs[n].imshow(imgs[n])
        axs[n].axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * show_size * 0.25)
    plt.savefig('save.png')
    plt.show()


def make_gif(imgs):
    plt.gray()
    fig, ax = plt.subplots(
        1, 1,
        gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.set_tight_layout(True)
    def images(i):
        ax.imshow(imgs[i])
        return ax
    anim = FuncAnimation(fig, images, frames=np.arange(len(imgs)), interval=500)
    anim.save('make.gif', dpi=80, writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    make_gif(imgs)
    show_images(imgs)