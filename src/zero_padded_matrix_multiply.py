# Author : Ali Snedden
# Date   : 12/28/18
# Purpose: 
#   I'm having difficulty getting matrix_multiply_cache_opt_tensor.cu to 
#   output the correct values using the tensor cores. I'm going to check
#   that I'm not in fact crazy and see that I can get the right values
#   when using the same arrays with numpy.
#
#
#
import numpy as np
import sys 


def exit_with_error(arg):
    sys.stderr.write(arg)
    sys.exit(1)


def main():
    if(len(sys.argv) != 3):
        exit_with_error("ERROR!! Wrong number of args. 2 paths expected to matrix files")
    aL = []
    bL = []

    ## Read files
    fileA = open(sys.argv[1])  # File A.txt
    for line in fileA:
        entries = line.strip().split()          
        entries = [float(x) for x in entries] ## Convert from str to float
        aL.append(entries)
    fileA.close()

    fileB = open(sys.argv[2])  # File B.txt
    for line in fileB:
        entries = line.strip().split()          
        entries = [float(x) for x in entries] ## Convert from str to float
        bL.append(entries)

    A = np.asarray(aL)
    B = np.asarray(bL)

    #print(np.dot(A,B)) ### Works to here
    # Now zero pad
    nZeroX = 16 - A.shape[0] % 16
    nZeroY = 16 - A.shape[1] % 16
    A = np.pad(A, ((0,nZeroX),(0,nZeroY)), 'constant')
    nZeroX = 16 - B.shape[0] % 16
    nZeroY = 16 - B.shape[1] % 16
    B = np.pad(B, ((0,nZeroX),(0,nZeroY)), 'constant')

    print(np.dot(A,B)) ### Works to here
    print("Finished")


if __name__ == "__main__":
    main()



