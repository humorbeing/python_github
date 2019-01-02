text = 'what ever i write here will end up\n in the file i open'

saveFile = open('text.txt','w') #'r','w','a'
#saveFile.write('\n')#if appending
saveFile.write(text)
saveFile.close()
readFile = open('text.txt','r').read()
#no close?
print (readFile)
readline = open('text.txt','r').readlines()
print(readline)
