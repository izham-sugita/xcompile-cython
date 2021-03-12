from PIL import Image

filename = input("Enter file name: ")

outfile = filename.replace('.png','-grey.png')

img = Image.open(filename).convert('LA')
img.save(outfile)
