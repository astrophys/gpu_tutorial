import random
import numpy as np
import time

def write_matrix(Matrix = None, FileName = None): 
    """
    ARGS:
        Matrix   : Matrix to write to filename
        FileName : 
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
    """
    f = open(FileName, "w+")
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            f.write("{} ".format(Matrix[i,j]))
        f.write("\n")
    f.close()

startTime = time.time()

random.seed(42)
A=np.zeros([200,10000])
B=np.zeros([10000,300])

for i in range(A.shape[0]):
  for j in range(A.shape[1]):
    A[i,j] = random.randint(0,10)

for i in range(B.shape[0]):
  for j in range(B.shape[1]):
    B[i,j] = random.randint(0,10)

AB = np.dot(A,B)
# 
# array([[  7.,   0.,   3., ...,   5.,  10.,   9.],
#        [  0.,   7.,   7., ...,   0.,  10.,   9.],
#        [ 10.,  10.,   9., ...,   0.,   0.,   9.],
#        ..., 
#        [  3.,   3.,   8., ...,   8.,   5.,   2.],
#        [  7.,   9.,   3., ...,   1.,   1.,  10.],
#        [  5.,   8.,  10., ...,   9.,   2.,   0.]])

# array([[ 10.,   9.,   7., ...,   8.,   9.,   9.],
#        [  0.,   1.,   2., ...,   8.,  10.,   1.],
#        [  1.,   9.,   5., ...,   0.,   8.,   2.],
#        ..., 
#        [  7.,   4.,   7., ...,   4.,   7.,   3.],
#        [  3.,   7.,   9., ...,   6.,   3.,   1.],
#        [  5.,   6.,   2., ...,   1.,   7.,   5.]])
# 

write_matrix(Matrix=A, FileName="data/A.txt")
write_matrix(Matrix=B, FileName="data/B.txt")
write_matrix(Matrix=AB, FileName="data/AB.txt")

print("Ended : %s"%(time.strftime("%D:%H:%M:%S")))
print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))


