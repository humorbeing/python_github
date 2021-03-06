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

        reads = line.split(' ')

        step_line = reads[0]
        reward_line = reads[-1]
        # print(step_line)
        # print(reward_line)
        step_line = step_line.split(':')
        step_num = int(float(step_line[1]))
        # print(step_num)
        # print(step_num+1)

        # print(reward_line)
        reward_num = float(reward_line)
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
        # log.append([step_num, reward_num])

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
# plot_this('e1.txt', 'Nothing', num=1)
# plot_this('e2.txt', '6 action', num=20)
plot_this('e3.txt', '3 action', num=20)
plot_this('e4.txt', '3 action gamma', num=20)
plot_this('e5.txt', '6 action gamma', num=10)
plot_this('e6.txt', '3 action g,gae,v,a s', num=6)
plot_this('e01.txt', '3 action rnn', num=3)
plot_this('e02.txt', '3 action 32rnn', num=3)
plot_this('e03.txt', '3 action 64rnn', num=3)
# plot_this('en_load_nofreeze.txt', 'En noFreeze')
# plot_this('en_load_freeze.txt', 'En Freeze')
# plot_this('cl_load_good1_freeze.txt', 'CL g1 Freeze')
# plot_this('cl_g1_noFre.txt', 'CL g1 noFreeze', 'r', 30)
# plot_this('cl_g2_noFre.txt', 'CL g2 noFreeze', 'b', 2)
# plot_this('RNN_10000_limit.txt', 'RNN(10000)')
# plot_this('vae_rnn.txt', 'VAE')
ax.grid(True)
ax.legend(loc='upper left')
# ax.set_title('A3C Learning Log')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
ax.set_xlim(left=0, right=5000000*0.9)
# ax.set_ylim(bottom=-22, top=-6)
plt.show()