from package import mod
from package.subpackage import mod as submod
# __init__.py is only for python 2.xxx
# import sys
# sys.path.insert(0, '../')
# sys.path.insert(0, '../../')
mod.hi()
submod.hi()