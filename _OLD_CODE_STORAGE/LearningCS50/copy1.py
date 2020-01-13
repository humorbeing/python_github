
print("s: ", end="")
s = input()
if len(s) == 0:  #why s == None doesn't work?
    exit(1)

c = s.capitalize() #can't touch s, only give a copy.

print("s: {}".format(s))
print("c: {}".format(c))

exit(0)
