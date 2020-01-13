"""
delete all but
un-git: rm -Rf .git .gitignore
"""


import os
# from subprocess import call
# call(["ls", "-l"])
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
                # os.remove(target)
                try:
                    # print('deleting:', target)
                    os.remove(target)
                except OSError:
                    print('FAIL to delete:', target)
                    print("If it's a git folder, try command: [rm -Rf .git .gitignore] to un-git the folder.")

            else:
                print('Would be deleting:', target)
