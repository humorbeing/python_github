# openAI gym

- setuptools version: Using 39.2.0. If to use 40.6.3,
there will be some 'warning' or error
- uu.py: it seems using this name will engage a "AttributeError: module 'pkg_resources' has no attribute 'EntryPoint'".
    - Looks like, if there is uu.py in the same directory, this error will occur.
    - as long as there is no uu.py file, everything works fine. Maybe there is uu.py used in setuptools.