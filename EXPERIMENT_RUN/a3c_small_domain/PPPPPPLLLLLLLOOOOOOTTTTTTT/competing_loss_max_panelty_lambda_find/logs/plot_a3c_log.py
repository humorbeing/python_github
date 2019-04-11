import numpy as np
import matplotlib.pyplot as plt

def xy(name):
    with open(name) as f:
        lines = [line for line in f]


    log = []
    count = 0
    # step_stack = []
    # reward_stack = []
    lss = []
    miss = []
    mass = []
    for line in lines:
        count += 1
        # if count % 500 == 0:

        reads = line.split(',')
        print(reads)

        reads = [cleanse.strip() for cleanse in reads]
        print(reads)
        # break
        loss_line = reads[1]
        mi_line = reads[2]
        max_line = reads[3]
        # print(step_line)

        # print(reward_line)
        loss_line = loss_line.split(' ')
        loss = float(loss_line[1])
        print(loss)
        mi_line = mi_line.split(' ')
        # print(mi_line)
        # break
        mi = float(mi_line[2])
        print(mi)
        max_line = max_line.split(' ')
        # print(mi_line)
        # break
        ma = float(max_line[2])
        print(ma)
        lss.append(loss)
        miss.append(mi)
        mass.append(ma)
    return lss, miss, mass
name = '20190410-14-26-18-WM_competing_loss_lambda_0.5.txt'
# name = '20190410-14-26-38-WM_competing_loss_lambda_0.7.txt'
# name = '20190410-14-26-49-WM_competing_loss_lambda_0.2.txt'
name = '20190410-15-14-30-WM_competing_loss_lambda_0.85_more_E.txt'
# name = '20190410-15-14-12-WM_competing_loss_lambda_0.75_more_E.txt'
name = '20190410-16-14-07-WM_competing_loss_more_layer_lambda_0.7.txt'
name = '20190410-16-15-28-WM_competing_loss_more_layer_lambda_0.85.txt'
l, i, x = xy(name)
# print(l)



def plot_this(file_name, plot_name):
    x, y = xy(file_name)
    ax.plot(x, y, label=plot_name)


# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
ax.plot(l, label='loss')
ax.plot(i, label='mi loss')
ax.plot(x, label='ma loss')
# plot_this('3ss.txt', '3 Simp')
# plot_this('max.txt', 'Max1')
# plot_this('rnn.txt', 'RNN')
# plot_this('ex.txt', 'RNN(EXT)')
# plot_this('r100.txt', 'Rd1')
# plot_this('r1000.txt', 'Rd2')
# plot_this('max2.txt', 'Max2')
# plot_this('vae_rnn.txt', 'VAE')
ax.grid(True)
ax.legend(loc='upper left')
ax.set_title('competing loss, More Layer, lambda=0.85')
ax.set_xlabel('epoch')
ax.set_ylabel('loss value')
# ax.set_xlim(left=0, right=50000000*1)
# ax.set_ylim(bottom=-22, top=-10)
plt.show()
