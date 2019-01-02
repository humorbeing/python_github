import sys
sys.path.insert(0, '../')
from package.mod import model
from package.mod import get_im_var
from package.mod import get_output
from package.mod import get_order
from package.mod import get_string

# import torch
# torch.nn.Module.dump_patches = True
import numpy as np

cnn, classes, performance = model('all')

inputs = get_im_var()
y = get_output(cnn, inputs)
print(y)
s = get_order(y)
print(s)
whole = get_string(classes, performance, y, s)
print(whole)
f = open('result.txt', 'w')
for l in whole:
    for w in l:
        f.write(w+' ')
    f.write('\n')
f.close()
