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
plt.title('Reconstruction, sigmoid, Train with MSE vs BCE, show MSE loss')
train_mark = '-'
val_mark = ':'


name = '20200111-200102-O_Adm-M1.1-NM-Zf-B_h256.txt'
title = '256'
c = 'r'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[5], c+val_mark, label=title+' Va')
#
name = '20200111-212001-O_Adm-M1.1-NM-Zf-B.txt'
title = 'normal'
c = 'g'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[5], c+val_mark, label=title+' Va')

name = '20200112-060957-O_Adm-M1.1-NM-Zf-B_h512.txt'
title = '512'
c = 'b'
a = xy(name)
plt.plot(a[1], c+train_mark, label=title+' Tr')
plt.plot(a[5], c+val_mark, label=title+' Va')
# #
# name = '20200111-212001-O_Adm-M1.1-NM-Zf-B.txt'
# title = 'nothing'
# c = 'y'
# a = xy(name)
# plt.plot(a[1], c+train_mark, label=title+' Tr')
# plt.plot(a[5], c+val_mark, label=title+' Va')
# # #
# name = '20200111-140858-O_Adm-M1.1-100M-Zf-B.txt'
# title = '100 M Zf B M1.1 1'
# c = 'c'
# a = xy(name)
# plt.plot(a[1], c+train_mark, label=title+' Tr')
# plt.plot(a[5], c+val_mark, label=title+' Va')
# # #
# name = '20200111-212001-O_Adm-M1.1-NM-Zf-B.txt'
# title = 'N M Zf B M1.1'
# c = 'm'
# a = xy(name)
# plt.plot(a[1], c+train_mark, label=title+' Tr')
# plt.plot(a[5], c+val_mark, label=title+' Va')
# # #
# name = '20200113-081629-O_Adm-M1.1-100M-Zf-B.txt'
# title = '100 M Zf B M1.1'
# c = 'k'
# a = xy(name)
# plt.plot(a[1], c+train_mark, label=title+' Tr')
# plt.plot(a[5], c+val_mark, label=title+' Va')

# name = '20200109-092304-0sb-zf-recon.txt'
# title = 'B Zf R'
# c = '#641E16'
# a = xy(name)
# plt.plot(a[1], color=c, linestyle=train_mark, label=title+' Tr')
# plt.plot(a[5], color=c, linestyle=val_mark, label=title+' Va')
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