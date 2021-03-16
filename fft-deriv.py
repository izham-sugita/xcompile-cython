import numpy as np

N = 100
xmin = 0.0
xmax = 2.0*np.pi
step = (xmax - xmin)/float(N)
xdata = np.linspace(step, xmax, N)

v = np.sin(xdata)

#analytical derivative
dvdx = np.cos(xdata)

#fft for v
vhat = np.fft.fft(v)

#calculating derivative from fft
what = 1j*np.zeros(N)
idhalf = int(N/2)
what[0:idhalf] = 1j*np.arange(0,idhalf,1)
what[idhalf+1:] = 1j*np.arange(-idhalf+1,0,1)
what = what*vhat
w = np.real( np.fft.ifft(what))

import matplotlib.pyplot as plt

plt.plot(xdata,dvdx,'-.b',linewidth=4,label='Analytical')
plt.plot(xdata, w, "-r", label="FFT")
plt.legend(loc="upper left")


plt.show()
