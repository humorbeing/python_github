'''
U0 = []
Us(l)
Us(l,kl,)
Us(l,kl,n) #f
Uc(l,kl,n) #f
Vc(l,n) #f
Vs(l,n) #f
Fy(x) #f
PHy(x) #f x+a != 0, or return Fy(x)
r(l) #f
k(l) #f
al(kl-1,v,kl)


'''

def Us(l_th,k_l_th,input):
    EE = 0
    for i in range(K(l_th-1)):
        for
    #r(l_th)*Fy((1+EEa()*Uc())/(1+()*b()*Vc())-1)
    return output

def Uc(l_thi,k_l_th,input):
    return output

def Vc(l_th,input):
    return output

def Vs(l_th,input):
    return output

def Fy(x):
    if x>=0:
        return x
    else:
        return 0

def Phy(x):
    alpha = 5
    if (alpha+x) != 0:
        return Fy(x/(alpha+x))
    else:
        print("Phy have 0 div problem!,return X")
        return Fy(x)

def r(l):
    r1 = 5
    r2 = 6
    r3 = 7
    if l == 1:
        return r1
    elif l == 2:
        return r2
    elif l == 3:
        return r3
    else:
        print("wrong r(l) input,defualt is {}".format(r1))
        return r1

def k(l):
    k0 = 1
    k1 = 5
    k2 = 6
    k3 = 7
    if l == 0:
        return k0
    elif l == 1:
        return k1
    elif l == 2:
        return k2
    elif l == 3:
        return k3
    else:
        print("wrong k(l) input,return defualt is 1")
        return 1
