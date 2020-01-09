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
plt.title('Reconstruction, Zero input, No args')
train_mark = '-'
val_mark = ':'
name = '20191228-12-52-23-lstm_encoder_decoder.txt'
name = '20200101-20-39-27-cnn_flatten_en_de_recon.txt'
title = 'CNN flatten'
c = 'r'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '20191228-19-59-29-lstmcell_encoder_decoder.txt'
title = 'LSTMCell'
c = 'g'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

# name = '20200102-01-48-30-copy_de_ori_recon.txt'
name = '20191229-03-29-43-cnn_lstmcell_encoder_decoder.txt'
title = 'CNN seed1'
c = 'b'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '20191229-08-27-41-cnn_lstmcell_encoder_decoder.txt'
title = 'CNN seed2'
c = 'y'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '20191231-07-15-21-cnn_lstmcell_v0002_encoder_decoder.txt'
title = 'CNN v2'
c = 'c'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '20200101-13-20-07-lstmcell_colab_en_de_recon.txt'
title = 'colab'
c = 'm'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[2], c+val_mark, label=title+' Va')

name = '20200101-19-39-19-cnn_v0003_en_de_recon.txt'
title = 'CNN v3'
c = 'k'
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