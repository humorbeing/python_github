import numpy as np

ar = np.random.random(10)
print(ar)

sd = np.std(ar)
print(sd)

var = np.var(ar)

print(var)

# sd_from_var = np.sqrt(var/len(ar))  # not this. sd = sqrt(var)
# actually np.sqrt(var/len(ar)) is standard deviation for
# t or z test for observation of the samples.
sd_from_var = np.sqrt(var)
print(sd_from_var)

def aa(i):

    print(i)

def bb(i,j):
    print(i+j)

aa(5)
# print(aa())


dd = {1:aa,
      2:bb,
      }

dd[1](6)
dd[2](1,2)

a = [1,2]

# print(a*0.5)
a = np.array(a)
print(a*0.5)

for _ in range(50):
    alpha = np.random.uniform(size=2)
    print(alpha)
    beta = np.array([1, 1])
    beta = beta - alpha
    print('b',beta)

a = np.array([1,2,3,4,5])
print(1-a)
