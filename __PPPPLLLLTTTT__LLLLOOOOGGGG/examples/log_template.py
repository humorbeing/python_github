

# separate ','
# aaa_a: 333

# aa_aa: 333, bb_bbb: 4.5, cc_ccc: 0.0001
# first separate with ','
# then separate with ' '
# then get the last of this separation

s = 'aa_aa: 333, bb_bbb: 4.5, cc_ccc: a.0001'
segs = s.split(',')
segs = [cleanse.strip() for cleanse in segs]
for i in range(len(segs)):
    all_s = segs[i].split(' ')

    print(all_s)
    num_s = all_s[-1]
    try:
        num = float(num_s)
    except:
        num = 'segment {} is not number'.format(i+1)
        print(num)