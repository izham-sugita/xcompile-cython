import numpy as np

N = 1000

a = np.random.rand(N,N)

header = str(N)+" "+str(N)+"\n"
fw = open("test.txt","w")
fw.write(header)
fw.write(str(a))
fw.write("\n")
fw.close()

fr = open("test.txt","r")
text = fr.read(3) #read until 3 space, data is string type
#print(text)
fr.close()

'''
print("Read line by line")
with open("test.txt","r") as fr:
    while True:
        line = fr.readline()
        if not line:
            break
        print(line)
'''

del fr
del fw

#sample matrix
b = np.ndarray((N,N))
b[:][:] = a[:][:]

#print(b)
print(type(b))

b = b.reshape(N*N)
#print(b)

c = b.reshape(N,N)
#print(c)

print("Write to text array")
fw = open("array.txt","w")
for i in range(N):
    for j in range(N):
        fw.write( str(c[i][j])+"\n" )

fw.close()

print("Read write binary file")
fw = open("array.bin","wb")
fw.write(c)
fw.close()









