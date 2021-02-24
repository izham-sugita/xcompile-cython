import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

#adding scipy for sparse matrix
from scipy.sparse import csr_matrix, save_npz


#img = mpimg.imread('sample0.png')
img = mpimg.imread('newSample-old.png')

imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

oimax, ojmax = imgB.shape
print("Image before fft: ", oimax, ojmax)

#filtering with 2D-fft
Bfft = np.fft.rfft2(imgB)
Bmagnitude = np.sqrt( Bfft[:][:].real**2 + Bfft[:][:].imag**2 )
Bmax = np.amax(Bmagnitude)
print(Bmax)
Bfilter = Bfft

imax, jmax = Bfft.shape
print("Image after fft: ", imax, jmax)

for i in range(Bfft.shape[0]):
    for j in range(Bfft.shape[1]):
        coeff = np.sqrt(Bfilter[i][j].real**2 + Bfilter[i][j].imag**2 )
        if(coeff < 0.00008*Bmax):
            Bfilter[i][j] = 0.0


zeros = (Bfilter == 0.0).sum()
tot_pix = imax*jmax
sparsity = float(zeros)/float(tot_pix)
print("Compressed image sparcity: ", sparsity)

sparseB = csr_matrix(Bfilter)
#print(sparseB)

#original image
zeros = (imgB == 0.0).sum()
imax, jmax = imgB.shape
print(imax,jmax)
tot_pix = imax*jmax
sparsity = float(zeros)/float(tot_pix)
print("Original image sparcity: ", sparsity)


IBfilter = np.fft.irfft2(Bfilter)
fimax, fjmax = IBfilter.shape
sparseIB = csr_matrix(IBfilter)
print(fimax,fjmax)

#plot side by side
fig = plt.figure()

b = fig.add_subplot(1,2,1)
imgplot = plt.imshow(imgB)
imgplot.set_clim(0.0,0.7)
b.set_title('Original Image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

b = fig.add_subplot(1,2,2)
imgplot = plt.imshow(IBfilter)
imgplot.set_clim(0.0,0.7)
b.set_title('FFT/Lossy Image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

#plt.imshow(IBfilter)

plt.show()

