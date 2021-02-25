import numpy as np

N = 3

a = np.random.rand(N,N)

#sample matrix
b = np.ndarray((N,N))
b[:][:] = a[:][:]

print(type(b))

b = b.reshape(N*N)
c = b.reshape(N,N)

print("Write to text array")
fw = open("array.txt","w")
for i in range(N):
    for j in range(N):
        fw.write( str(c[i][j])+"\n" )
 
fw.close()

print("Write binary file")
fw = open("array.bin","wb")
fw.write(c)
fw.close()

print("Read binary file")
fr = open("array.bin","rb")
c = fr.read()
fr.close()
print(type(c))
#print(c)


#1D array
#x = np.array([1.0,2.0,3.0])

#2D array
#x = np.random.rand(2,2)
x = np.random.rand(N,N)

fname ="dummy"
#np.save will produce a system independent binary file .npy
np.save(fname,x)
del x

x = np.load(fname +".npy")
print(type(x))
print(x)







