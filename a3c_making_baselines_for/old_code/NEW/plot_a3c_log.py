import numpy as np
import matplotlib.pyplot as plt

def xy(name):
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
        if count % 40 == 0:
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


def plot_this(file_name, plot_name):
    x, y = xy(file_name)
    ax.plot(x, y, label=plot_name)


# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
plot_this('encoder_rnn.txt', 'ENCODER')
plot_this('mlp.txt', 'MLP')
plot_this('more_rnn.txt', 'RNN(More)')
plot_this('NO_limit_rnn.txt', 'RNN(NOlim)')
plot_this('pix_pix.txt', 'Pixel')
plot_this('pix_rnn.txt', 'RNN(Pixel)')
plot_this('RNN_10000_limit.txt', 'RNN(10000)')
plot_this('vae_rnn.txt', 'VAE')
ax.grid(True)
ax.legend(loc='lower right')
ax.set_title('A3C Learning Log')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
ax.set_xlim(left=0, right=50000000*0.5)
# ax.set_ylim(bottom=-22, top=21)
plt.show()