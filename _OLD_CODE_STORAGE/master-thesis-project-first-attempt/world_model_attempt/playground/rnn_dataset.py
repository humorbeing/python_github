"""
description of code
"""

from this_utility import *
from vae import VAE

is_cuda = True
vae_model = VAE()
vae_model.load_state_dict(torch.load(vae_model_path, map_location=lambda storage, loc: storage))
vae_model.eval()
if is_cuda:
    vae_model.cuda()
    vae_model.is_cuda = True
filelist = os.listdir(DATA_DIR)
filelist.sort()
N = len(filelist)

z_list = []
action_list = []
for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data = raw_data['obs']
    data = torch.from_numpy(data)
    if is_cuda:
        data = data.cuda()
    t = vae_model(data)
    if is_cuda:
        t = t.cpu()
    z = t.data.numpy()
    z_list.append(z)
    action_list.append(raw_data['action'])

    if ((i + 1) % 10 == 0):
        print("loading file", (i + 1))

z_list = np.array(z_list)
action_list = np.array(action_list)
np.savez_compressed(
    os.path.join(SERIES_DIR, "series3.npz"),
    action=action_list,
    z=z_list)

'''
series:
 series.npz all of them 8000
 series1.npz all of part of 8000
 series2.npz 20
 series3.npz 1
'''
