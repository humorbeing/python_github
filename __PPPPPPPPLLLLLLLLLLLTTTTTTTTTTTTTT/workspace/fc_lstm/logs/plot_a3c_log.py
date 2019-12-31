import numpy as np
import matplotlib.pyplot as plt

def xy(name):
    with open(name) as f:
        lines = [line for line in f]


    log = []
    count = 0
    x1 = []
    x2 = []
    for line in lines:
        count += 1
        # if count % 500 == 0:

        reads = line.split(',')
        reads = [cleanse.strip() for cleanse in reads]
        # print(reads)
        a1 = reads[1]
        a2 = reads[2]
        # print(step_line)
        # print(reward_line)
        step_line = a1.split(' ')
        # print(step_line)
        step_num = float(step_line[2])
        # print(step_num)
        x1.append(step_num)
        step_line = a2.split(' ')
        # print(step_line)
        step_num = float(step_line[2])
        # print(step_num)
        x2.append(step_num)


    return x1, x2
name = '20191228-12-52-23-lstm_encoder_decoder.txt'


def plot_this1(file_name, plot_name):
    x, y = xy(file_name)
    # print(x)
    # print(len(x))
    ax.plot(range(len(x)),x, label=plot_name)

def plot_this2(file_name, plot_name):
    y, x = xy(file_name)
    # print(x)
    # print(len(x))
    ax.plot(range(len(x)),x, label=plot_name)

# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
plot_this1(name, 'ENCODER')
plot_this2(name, 'ENCODER')
# plot_this('mlp.txt', 'MLP')
# plot_this('more_rnn.txt', 'RNN(More)')
# plot_this('NO_limit_rnn.txt', 'RNN(NOlim)')
# plot_this('pix_pix.txt', 'Pixel')
# plot_this('pix_rnn.txt', 'RNN(Pixel)')
# plot_this('RNN_10000_limit.txt', 'RNN(10000)')
# plot_this('vae_rnn.txt', 'VAE')
ax.grid(True)
ax.legend(loc='upper left')
ax.set_title('A3C Learning Log')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
# ax.set_xlim(left=0, right=50000000*0.4)
# ax.set_ylim(bottom=-22, top=-10)
plt.show()