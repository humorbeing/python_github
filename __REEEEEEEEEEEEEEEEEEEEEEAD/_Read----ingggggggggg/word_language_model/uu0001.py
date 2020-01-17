

# a = Dictionary()
import argparse
a = argparse.Namespace()
a.b = 1
print(getattr(a, 'b'))
def c(d):
    print(d)
    return 'e'
a.b = c
print(getattr(a, 'b')(5))
'''
'''