import numpy as np
import matplotlib.pyplot as plt

def xy(name):
    with open(name) as f:
        lines = [line for line in f]
    num_logs = 0

    all_num = []
    for line in lines:
        num_from_line = []
        num_logs += 1
        segments = line.split(',')
        segments = [cleanse.strip() for cleanse in segments]
        for i in range(len(segments)):
            seg_name_num = segments[i].split(' ')
            num_s = seg_name_num[-1]
            try:
                num = float(num_s)
            except:
                num = 0.0
                print('segment {} is not number'.format(i + 1))
            num_from_line.append(num)
        all_num.append(num_from_line)
    return np.array(all_num).T
# plt.plot(y, 'ro', label='This is y')
plt.figure(figsize=(6.4, 4.8), dpi=100)
train_mark = '-'
val_mark = ':'
name = '1.txt'
title = 'LSTM'
c = 'r'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '2.txt'
title = 'CNN'
c = 'g'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')




''' legend location
center left
center right
lower left
best
lower right
upper left
lower center
upper right
upper center
center
right
'''

# plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
# plt.legend(bbox_to_anchor=(0.9, 0.6))

plt.xlabel('Frame')
plt.ylabel('Episodic Reward')
# plt.xlim(left=0, right=50000000*0.4)
# plt.ylim(bottom=-22, top=-10)

plt.show()









''' Line style
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
'1'	tri_down marker
'2'	tri_up marker
'3'	tri_left marker
'4'	tri_right marker
's'	square marker
'p'	pentagon marker
'*'	star marker
'h'	hexagon1 marker
'H'	hexagon2 marker
'+'	plus marker
'x'	x marker
'D'	diamond marker
'd'	thin_diamond marker
'|'	vline marker
'_'	hline marker
'''

''' Colors
‘b’	blue
‘g’	green
‘r’	red
‘c’	cyan
‘m’	magenta
‘y’	yellow
‘k’	black
‘w’	white
'''