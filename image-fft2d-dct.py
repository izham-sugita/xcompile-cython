import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.fftpack import dct, idct
from scipy.io import mmwrite, mmread
from PIL import Image

#adding scipy for sparse matrix
from scipy.sparse import csr_matrix, save_npz, load_npz

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

#img = mpimg.imread('sample0.png')
#img = mpimg.imread('newSample-old.png')
img = mpimg.imread('pin.png')
#img = mpimg.imread('lena_color_512.png')


imgR = img[:,:,0]
imgG = img[:,:,1]
imgB = img[:,:,2]

oimax, ojmax = imgB.shape
print("Image before fft: ", oimax, ojmax)
print("Size in bytes of the original image: ", sys.getsizeof(imgB), " bytes")

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

IBfilter = np.fft.irfft2(Bfilter)
sparseFFT = csr_matrix(Bfilter)
ifftsparse, jfftsparse = sparseFFT.shape
print(ifftsparse,jfftsparse)

Gdct = dct2(imgB)
print("DCT shape")
print(Gdct.shape)
imaxG, jmaxG = Gdct.shape

#filtering based on amplitude strength
#too arbirary

Gdctmag = np.sqrt( Gdct[:][:]**2 )
Gdctmax = np.amax( Gdctmag )
print(Gdctmax)
amp = []
for i in range(imaxG):
    for j in range(jmaxG):
        amp.append( np.sqrt( Gdct[i][j]**2 ) )

amp = sorted(amp, reverse=True)
half_j = int(jmaxG)
threshold_amp = int( len(amp)/10 ) #10%from the amplitude
ind_min_amp = amp[threshold_amp]
for i in range(imaxG):
    for j in range(jmaxG):
        if np.sqrt( Gdct[i][j]**2 ) < ind_min_amp:
            Gdct[i][j] = 0.0



#filtering based on frequency
#produced rippled image
'''
half_j = int(jmaxG/2)
quat_j = int(jmaxG/4)
for i in range(imaxG):
    for j in range(quat_j , jmaxG):
        Gdct[i][j] = 0.0
'''


sparseDCT = csr_matrix(Gdct)

mmwrite('my_array', sparseDCT)
mmformat = mmread('my_array').tolil()  
mmformat = mmformat.todense()

save_npz('sparseDCT.npz', sparseDCT)
sparse_matrix = load_npz('sparseDCT.npz')
sparse_matrix = sparse_matrix.todense()

#IGdct = idct2(Gdct)
#IGdct = idct2(sparse_matrix)
IGdct = idct2(mmformat)


zeros = (Gdct == 0.0).sum()
tot_pix = imaxG*jmaxG
sparsity = float(zeros)/float(tot_pix)
print("Compressed DCT image sparcity: ", sparsity)

zeros = (Bfilter == 0.0).sum()
tot_pix = imax*jmax
sparsity = float(zeros)/float(tot_pix)
print("Compressed FFT image sparcity: ", sparsity)





#plot side by side
fig = plt.figure()

b = fig.add_subplot(1,3,1)
imgplot = plt.imshow(imgB)
imgplot.set_clim(0.0,0.7)
b.set_title('Original Image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

b = fig.add_subplot(1,3,2)
imgplot = plt.imshow(IBfilter)
imgplot.set_clim(0.0,0.7)
b.set_title('FFT/Lossy Image')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

b = fig.add_subplot(1,3,3)
imgplot = plt.imshow(IGdct)
imgplot.set_clim(0.0,0.7)
b.set_title('DCT')
plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

#plt.imshow(IBfilter)

#Image.fromarray(IBfilter).save('compressed.tif')
#Image.fromarray(imgB).save('original.tif')


plt.show()

