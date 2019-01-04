#!/bin/env python
# Author : Ali Snedden
# License: MIT
# Date   : 1/3/19
# Purpose: 
#   To read in two files  and compare the values with some tolerance.
#
#
#
#
import sys
import time
import numpy as np

def exit_with_error(String):
    """
    ARGS:
    RETURN:
    DESCRIPTION:
    DEBUG:
    FUTURE:
    """
    sys.stderr.write("{}".format(String))
    sys.exit(1)


def print_help(arg):
    """
    ARGS:
        arg     : exit value
    DESCRIPTION:
        Print Help. Exit with value arg
    RETURN:
        N/A
    DEBUG:
        1. Tested, it worked
    FUTURE:
    """
    sys.stdout.write(
        "\nUSAGE : ./output_cmp.py file1.txt file2.txt tol \n\n"
        "            tol : Tolerance.  A float between [0,1]\n")
    sys.exit(arg)



def read_file(Path):
    """
    ARGS:
        Path : path to file
    RETURN:
        numpy array
    DESCRIPTION:
    DEBUG:
        1. Tested Error
        2. Spot checked. Works
    FUTURE:
    """
    fin = open(Path, "r")
    nLines  = 0
    nNum    = 0          # Current number of numbers per line
    nNumPrev= 0          # Current number of numbers per line
    firstRead= True
    ## Get file dimensions ##
    for line in fin:
        line = filter(None, line.strip().split(' '))

        ## Check for errors
        if(firstRead == True):
            nNum = len(line)
            nNumPrev = len(line)
            firstRead = False
        else:
            nNum = len(line)

        if(nNum != nNumPrev):
            exit_with_error("ERROR!!! Formatting problems in {} : nNum ({}) != "
                            "nNumPrev ({})\n".format(Path,nNum,nNumPrev))
        nNumPrev = nNum
        nLines = nLines + 1
    fin.close()
    fin = open(Path, "r")
    print("Reading {} ... [{}, {}]\n".format(Path,nLines,nNum))

    ## Allocate memory ##
    outM = np.zeros([nLines,nNum])      ## Output Matrix
    i = 0   # Row index
    j = 0   # Column index
    for line in fin:
        line = filter(None, line.strip().split(' '))
        outM[i,:] = np.asarray(line, dtype=float)
        #for num in line:
        #    try: 
        #        outM[i,j] = float(num)
        #    except ValueError:
        #        import pdb
        #        import traceback
        #        extype, value, tb = sys.exc_info()
        #        traceback.print_exc()
        #        pdb.post_mortem(tb)
        #    j = j + 1
        #j = 0
        i = i + 1
    fin.close()
    
    return(outM)



def main():
    """
    ARGS:
    RETURN:
    DESCRIPTION:
    DEBUG:
    FUTURE:
    """
    ######### Get Command Line Options ##########
    if(len(sys.argv) != 4):
        print_help(1)
    elif(sys.argv[1] == "--help" or sys.argv[1] == "-help"):
        print_help(0)
    print("Started : %s"%(time.strftime("%D:%H:%M:%S")))
    startTime = time.time()
    tol = float(sys.argv[3])

    ### I/O ###
    m1 = read_file(sys.argv[1])     # matrix from file 1
    m2 = read_file(sys.argv[2])     # matrix from file 2
    
    ### Check for off Dimensions ###
    if(m1.shape != m2.shape):
        exit_with_error("ERROR!!! Incorrect dimensions {} != {}\n".format(m1.shape, m2.shape))

    ### Check for within tolerance ###
    isClose = np.allclose(m1, m2, tol)
    if(isClose == True):
        print("IDENTICAL :: {} and {} are within {} fractional tolerance\n".format(
              sys.argv[1], sys.argv[2], tol))
    else:
        import pdb
        import traceback
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        nDiff = m1.shape[0] * m1.shape[1] - np.count_nonzero(np.isclose(a=m1,b=m2,rtol=tol))
        print("DIFFERENT :: {} and {} not within {} tol. {}/{} = {:<.2f} %  "
              "different \n".format(sys.argv[1], sys.argv[2], tol, nDiff,
              m1.shape[0] * m1.shape[1], nDiff/float(m1.shape[0]*m1.shape[1])*100))
        

    print("Finished in {:.4} h".format(
         (time.time()- startTime)/3600.0))


if __name__ == "__main__":
    main()
