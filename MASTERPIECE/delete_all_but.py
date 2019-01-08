"""
delete all but
"""


import os

# for f in os.listdir('.'):
#     # if not f.endswith('.pdf'):
#     #     os.remove(f)
#     print(f)
#     # print(f.endswith('.py'))
# print('---  '*20)
# for root, dirs, files in os.walk('.'):
#     for f in files:
#         print(f)
REAL = True
# REAL = False

for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            pass
        elif f.endswith('.md'):
            pass
        elif f.endswith('.ipynb'):
            pass
        else:
            target = os.path.join(root, f)
            if REAL:
                print('deleting:', target)
                os.remove(target)
            else:
                print('Would be deleting:', target)
