import numpy as np

N = 10

#1D array
#x = np.array([1.0,2.0,3.0])

#2D array
#x = np.random.rand(2,2)
x = np.random.rand(N,N,N)

#save to text format first


fname ="dummy"
#np.save will produce a system independent binary file .npy
np.save(fname,x)
del x

x = np.load(fname +".npy")
print(type(x))
print(x)







