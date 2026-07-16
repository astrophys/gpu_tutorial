import sys
import time
import random
import argparse
import numpy as np

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


def print_help(arg):
    """
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES:
    FUTURE:
    """
    sys.stdout.write(
        "\nUSAGE : ./matrix_generator.py Ax Ay Bx By\n\n"
        "     Ax : dimension of A in x\n"
        "     Ay : dimension of A in y\n"
        "     Bx : dimension of B in x\n"
        "     By : dimension of B in y\n\n"
        )
    sys.exit(arg)



def main():
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
    parser = argparse.ArgumentParser(
                 description="This generates plots from output of `sacct`")
    parser.add_argument('--Ax', metavar='size_of_Ax', type=int,
                     help='Size x-dimension of A matrix')
    parser.add_argument('--Ay', metavar='size_of_Ay', type=int,
                     help='Size y-dimension of A matrix')
    parser.add_argument('--Bx', metavar='size_of_Bx', type=int,
                     help='Size x-dimension of B matrix')
    parser.add_argument('--By', metavar='size_of_By', type=int,
                     help='Size y-dimension of B matrix')
    parser.add_argument('--outpath', metavar='path_to_output_files', type=str,
                     help='Path to output files')
    startTime = time.time()
    random.seed(42)
    args = parser.parse_args()
    ######### Get Command Line Options ##########
    #if(len(sys.argv) != 5):
    #    print_help(1)
    #elif(sys.argv[1] == "--help" or sys.argv[1] == "-help"):
    #    print_help(0)
    Ax = args.Ax
    Ay = args.Ay
    Bx = args.Bx
    By = args.By
    outpath = args.outpath
    #A=np.zeros([2000,10000])
    #B=np.zeros([10000,3000])
    A=np.zeros([Ax,Ay])
    B=np.zeros([Bx,By])

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

    # array([[ 10.,   9.,   7., ...,   8.,   9.,   9.], #        [  0.,   1.,   2., ...,   8.,  10.,   1.],
    #        [  1.,   9.,   5., ...,   0.,   8.,   2.],
    #        ...,
    #        [  7.,   4.,   7., ...,   4.,   7.,   3.],
    #        [  3.,   7.,   9., ...,   6.,   3.,   1.],
    #        [  5.,   6.,   2., ...,   1.,   7.,   5.]])
    #

    write_matrix(Matrix=A, FileName=f'{outpath}/A.txt')
    write_matrix(Matrix=B, FileName=f'{outpath}/B.txt')
    write_matrix(Matrix=AB, FileName=f'{outpath}/AB.txt')

    print("Ended : %s"%(time.strftime("%D:%H:%M:%S")))
    print("Run Time : {:.4f} h".format((time.time() - startTime)/3600.0))

if __name__ == "__main__":
    main()

