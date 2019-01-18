from package import mod
from package.subpackage import mod as submod
"""
new way
"""
# from .package import mod
# from .check_out import a
'''
even tho it doesn't work, but you can actually 
pick inside of the imported package,
which is huge benefit
'''

'''
- when coding, use relative import,
easier to pick inside a module.
- when running code, change to normal way.
'''
# from .package import mod
# __init__.py is only for python 2.xxx
# import sys
# sys.path.insert(0, '../')
# sys.path.insert(0, '../../')
mod.hi()
submod.hi()