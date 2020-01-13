import pickle
xxx = 1
pickle.dump(xxx, open("xxx.save", "wb"))
xxxx = pickle.load(open("xxx.save", "rb"))
print(xxxx)