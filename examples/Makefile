all : ex1 ex2 ex3

debug = 0

CXXFLAGS = -std=c++17 -O3
#CUDAFLAGS = -Xptxas -dlcm=ca
ifeq ($(debug), 0)
CXXFLAGS += -DNDEBUG
endif

ifeq ($(debug), 1)
CXXFLAGS += -g
endif

ex1: ex1.cu
	nvcc ex1.cu -o ex1.x $(CUDAFLAGS) -Xcompiler $(CXXFLAGS) -lcurand

ex2: ex2.cu
	nvcc ex2.cu -o ex2.x $(CUDAFLAGS) -Xcompiler $(CXXFLAGS)

ex3: ex3.cu
	nvcc ex3.cu -o ex3.x $(CUDAFLAGS) -Xcompiler $(CXXFLAGS)

clean:
	rm -rf *.x

