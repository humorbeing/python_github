from PIL import Image
im = Image.open('a.jpg')
# im = im.resize((80, 80))
print(im.format, im.size, im.mode)
im.show()
