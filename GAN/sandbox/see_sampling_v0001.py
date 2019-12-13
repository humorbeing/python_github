import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
def discrete_cmap(N, base_cmap=None):
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z
batch_size = 1000
z_id_ = np.random.randint(0,10,size=[batch_size])
z = gaussian_mixture(batch_size,2, label_indices=z_id_)
plt.figure(figsize=(8, 6))
plt.scatter(z[:, 0], z[:, 1], c=z_id_, marker='o', edgecolor='none',
            cmap=discrete_cmap(10, 'jet')
            )
plt.colorbar(ticks=range(10))
plt.grid(True)
axes = plt.gca()
axes.set_xlim([-4.5, 4.5])
axes.set_ylim([-4.5, 4.5])
plt.show()