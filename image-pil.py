from PIL import Image
import numpy as np

from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T,norm='ortho').T, norm='ortho')

#image = Image.open("pin-ori.jpg")
image = Image.open("lena_color_256.jpg")

data = np.asarray(image)
print(type(data))
print(data.shape)

row,col,layer = data.shape
print(row,col,layer)
imR = np.ndarray((row,col))
imG = np.ndarray((row,col))
imB = np.ndarray((row,col))
imR[:,:] = data[:,:,0]
imG[:,:] = data[:,:,1]
imB[:,:] = data[:,:,2]

imRnp = np.ndarray((row,col))
imGnp = np.ndarray((row,col))
imBnp = np.ndarray((row,col))
imAll = np.ndarray((row,col,layer))
print(imR)
print(np.amax(imR))
imRmax = np.amax(imR)
imGmax = np.amax(imG)
imBmax = np.amax(imB)
#modified to print using matplotlib
for i in range(row):
    for j in range(col):
        imRnp[i][j] = float(imR[i][j])/float(imRmax)
        imGnp[i][j] = float(imG[i][j])/float(imGmax)
        imBnp[i][j] = float(imB[i][j])/float(imBmax) 
        

#recombining will give the original image in numpy array data type
imAll[:,:,0] = imRnp[:,:]
imAll[:,:,1] = imGnp[:,:]
imAll[:,:,2] = imBnp[:,:]

#compress image with DCT
Rdct = dct2(imRnp)
Gdct = dct2(imGnp)
Bdct = dct2(imBnp)

#calculating magnitude
Ramp = []
Gamp = []
Bamp = []
for i in range(Rdct.shape[0]):
    for j in range(Rdct.shape[1]):
        Ramp.append(np.sqrt(Rdct[i][j]**2) )
        Gamp.append(np.sqrt(Gdct[i][j]**2) )
        Bamp.append(np.sqrt(Bdct[i][j]**2) )

Ramp = sorted(Ramp, reverse=True)
Gamp = sorted(Ramp, reverse=True)
Bamp = sorted(Ramp, reverse=True)

threshR = int( len(Ramp)/100 )
threshG = int( len(Gamp)/100 )
threshB = int( len(Bamp)/100 )

indR = Ramp[threshR]
indG = Gamp[threshG]
indB = Bamp[threshB]

#R-element filter
for i in range(Rdct.shape[0]):
    for j in range(Rdct.shape[1]):
        if np.sqrt( Rdct[i][j]**2 ) < indR:
            Rdct[i][j] = 0.0

#G-element filter
for i in range(Gdct.shape[0]):
    for j in range(Gdct.shape[1]):
        if np.sqrt( Gdct[i][j]**2 ) < indG:
            Gdct[i][j] = 0.0

#B-element filter
for i in range(Bdct.shape[0]):
    for j in range(Bdct.shape[1]):
        if np.sqrt( Bdct[i][j]**2 ) < indB:
            Bdct[i][j] = 0.0


iRdct = idct2(Rdct)
iGdct = idct2(Gdct)
iBdct = idct2(Bdct)

imAllFilt = np.ndarray((row,col,layer))
imAllFilt[:,:,0] = iRdct[:,:]
imAllFilt[:,:,1] = iGdct[:,:]
imAllFilt[:,:,2] = iBdct[:,:]


import matplotlib.pyplot as plt
#plot side by side
fig = plt.figure()

b = fig.add_subplot(1,3,1)
imgplot = plt.imshow(imAll)
imgplot.set_clim(0.0,0.7)
b.set_title('Original image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

b = fig.add_subplot(1,3,2)
imgplot = plt.imshow(imGnp)
imgplot.set_clim(0.0,0.7)
b.set_title('Image Green')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

b = fig.add_subplot(1,3,3)
imgplot = plt.imshow(imAllFilt)
imgplot.set_clim(0.0,0.7)
b.set_title('DCT filtered image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

plt.show()

#create Pillow image
#image2 = Image.fromarray(imR)
#image2.show()
#print(image.format)
#print(image.size)
#print(image.mode)
#image.show()
