# To compile :
#   ml load openmpi/gcc/64/4.1.2
#
#

# changed 7/9/2024
###### mpi_matrix_mult ######
CXXFLAGS = -Wall -g
NVCC = nvcc
PREFIX = src
#NVCCFLAGS = --compiler-options ${CXXFLAGS} -I${CPATH} -I${CUDA_INC_PATH}/include

OBJ_DIR = $(PREFIX)
#CU_OBJ = $(OBJ_DIR)/functions.o

###### hello world ######
OBJ_HELLO=$(OBJ_DIR)/hello.o
OBJ_MM=$(OBJ_DIR)/matrix_multiply.o

all: hello_gpu matrix_multiply

# Compile straight CUDA files
# -dc avoids error : ptxas fatal   : Unresolved extern function '_Z9d_map_idxiii'
$(OBJ_HELLO) : $(PREFIX)/hello_gpu.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ_MM) : $(PREFIX)/matrix_multiply.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

hello_gpu : $(OBJ_HELLO)
	$(NVCC) -o hello_gpu $(OBJ_HELLO)

matrix_multiply : $(OBJ_MM)
	$(NVCC) -o matrix_multiply $(OBJ_MM)


clean :
	rm $(OBJ_MPI) hello_gpu matrix_multiply 
