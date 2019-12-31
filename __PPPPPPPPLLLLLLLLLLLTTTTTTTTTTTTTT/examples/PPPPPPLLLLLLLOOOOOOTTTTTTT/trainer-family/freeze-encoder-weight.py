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
        # step_stack.append(step_num)
        # reward_stack.append(reward_num)
        log.append([step_num, reward_num])
    # print('num raw data', count)
    log = np.array(log)
    # print(log.shape)
    log = log[log[:, 0].argsort()]

        # if count > 5000:
        #     break

    # print(log)
    logs = []
    step_stack = []
    reward_stack = []
    if num is None:
        num = 50
    for count in range(len(log)):
        # print(log[count])
        step_stack.append(log[count][0])
        reward_stack.append(log[count][1])
        if count % num == 0:
            s = np.mean(step_stack)
            r = np.mean(reward_stack)
            logs.append([s, r])
            step_stack = []
            reward_stack = []

    log  = np.array(logs)
    # print(log.shape)
    # print(log)
    # log.sort(axis=0)

    # print(log.shape)
    # print(log.shape)
    # print(log)
    t_log = np.transpose(log)
    # print(t_log.shape)
    # print(t_log)
    x = t_log[0]
    y = t_log[1]
    return x, y


def plot_this(file_name, plot_name, color=None, num=None):
    x, y = xy(file_name, num=num)
    ax.plot(x, y, label=plot_name, color=color)

# def plot_these(file_names, plot_name, color=None, num=None):
#     xs =
#     ys =

# plt.plot(x, y)
# plt.scatter(t_log[0], t_log[1])
fig, ax = plt.subplots()
# plot_this('test_log.txt', 'A3C')
# plot_this('a3c-1.txt', 'A3C')
# plot_this('a3c-200.txt', 'A3C')
# plot_this('a3c-500.txt', 'A3C')
# plot_this('dmb-all.txt', 'DMB(Our)', 'r', 100)
# plot_this('a3c-all.txt', 'A3C(Baseline)', 'g', 60)

plot_this('dmb-freeze-all.txt', 'DMB(our), Freeze weight', 'r', num=60)
plot_this('a3c-1-fre.txt', 'A3C, Freeze Weight')

# plot_this('a3c-all2.txt', 'A3C2-all')
# plot_this('en-1.txt', 'Autoencoder', num=40)
plot_this('en-fre.txt', 'AutoEncoder, Freeze weight')
# plot_this('g1-1.txt', '1')

# plot_this('g1-2-fre.txt', 'g2')
# plot_this('g1-200.txt', '2')
# plot_this('g1-500.txt', '3')
# plot_this('g1-1000.txt', '4')
# plot_this('g2-1.txt', '5')
# plot_this('g2-1000.txt', '6')
# plot_this('soso-1.txt', '7')
# plot_this('soso-1-fre.txt', '1000A3C')
# plot_this('soso-200.txt', '8')
# plot_this('soso-500.txt', '9')
# plot_this('mb-1.txt', 'Encoder')
plot_this('mb-1-fre.txt', 'Model-Based, Freeze weight', 'y')
# plot_this('mb-1000.txt', 'Model-based', num=60)



ax.grid(True)
ax.legend(loc='upper left')
ax.set_title('Pong-ram-v0 (Freeze encoder weight)')
ax.set_xlabel('Frame')
ax.set_ylabel('Episodic Reward')
ax.set_xlim(left=0, right=5000000*4)
ax.set_ylim(bottom=-22, top=-16)
plt.show()