import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

img = mpimg.imread('sample0.png')

imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

#fig = plt.figure()
#a = fig.add_subplot(1, 2, 1)
#imgplot = plt.imshow(imgB)
#a.set_title('B-image')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
#a = fig.add_subplot(1, 2, 2)
#imgplot = plt.imshow(img)
#imgplot.set_clim(0.0, 0.7)
#a.set_title('Original')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

#filtering with 2D-fft
Bfft = np.fft.rfft2(imgB)
Bmagnitude = np.sqrt(Bfft[:][:].real**2)
Bmax = np.amax(Bmagnitude)
Bfilter = Bfft
print(Bfft.shape[0])
print(Bfft.shape[1])
imax, jmax = Bfft.shape

for i in range(Bfft.shape[0]):
    for j in range(Bfft.shape[1]):
        coeff = np.sqrt(Bfilter[i][j].real**2 + Bfilter[i][j].imag**2 )
        if(coeff < 0.0002*Bmax):
            Bfilter[i][j] = 0.0


zeros = (Bfilter == 0.0).sum()
tot_pix = imax*jmax
sparsity = float(zeros)/float(tot_pix)
print("Compressed image sparcity: ", sparsity)

#original image
zeros = (imgB == 0.0).sum()
imax, jmax = imgB.shape
tot_pix = imax*jmax
sparsity = float(zeros)/float(tot_pix)
print("Original image sparcity: ", sparsity)


IBfilter = np.fft.irfft2(Bfilter)


plt.imshow(IBfilter)
plt.show()

