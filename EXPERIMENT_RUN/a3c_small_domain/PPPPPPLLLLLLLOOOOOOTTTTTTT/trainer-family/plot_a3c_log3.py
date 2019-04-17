import numpy as np
import matplotlib.pyplot as plt

def xy(name, num=None):
    with open(name) as f:
        lines = [line for line in f]


    log = []
    count = 0
    step_stack = []
    reward_stack = []
    for line in lines:
        count += 1
        # if count % 500 == 0:

        reads = line.split(',')
        reads = [cleanse.strip() for cleanse in reads]
        step_line = reads[1]
        reward_line = reads[3]
        # print(step_line)
        # print(reward_line)
        step_line = step_line.split(' ')
        step_num = int(step_line[2])
        # print(step_num)
        # print(step_num+1)
        reward_line = reward_line.split(' ')
        # print(reward_line)
        reward_num = float(reward_line[2])
        # print(reward_num)
        # print(reward_num+0.2)
        step_stack.append(step_num)
        reward_stack.append(reward_num)
        if num is None:
            num = 40
        if count % num == 0:
            s = np.mean(step_stack)
            r = np.mean(reward_stack)
            log.append([s, r])
            step_stack = []
            reward_stack = []

        # if count > 5000:
        #     break

    # print(log)
    log  = np.array(log)
    # print(log.shape)
    t_log = np.transpose(log)
    # print(t_log.shape)
    # print(t_log)
    x = t_log[0]
    y = t_log[1]
    return x, y


def plot_this(file_name, plot_name, color=None, num=None):
    x, y = xy(file_name, num=num)
    ax.plot(x, y, label=plot_name, color=color)


# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
# plot_this('1000_normal.txt', '1000A3C', 'm')
plot_this('soso_.txt', 'Soso noFreeze')
# plot_this('en_load_freeze.txt', 'en Freeze')
plot_this('en_load_nofreeze.txt', 'en noFreeze')
# plot_this('g1_freeze.txt', 'g1 Freeze')
plot_this('en_noload_nofreeze_normal_train.txt', 'A3c')
# plot_this('1000_g1.txt', '1000CL g1 noFreeze', 'r')
# plot_this('1000_g2.txt', '1000CL g2 noFreeze', 'b')
plot_this('cl_g1_noFre.txt', 'CL g1 noFreeze')
# plot_this('cl_g2_noFre.txt', 'CL g2 noFreeze')
# plot_this('mb_fre.txt', 'M-B freeze')
plot_this('mb_nofre.txt', 'M-B no Freeze')
# plot_this('RNN_10000_limit.txt', 'RNN(10000)')
# plot_this('vae_rnn.txt', 'VAE')
ax.grid(True)
ax.legend(loc='upper left')
# ax.set_title('A3C Learning Log')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
ax.set_xlim(left=0, right=5000000*4)
ax.set_ylim(bottom=-22, top=-5)
plt.show()