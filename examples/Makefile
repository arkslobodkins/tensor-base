all : ex1 ex2 ex3 ex4

debug = 1

CXXFLAGS = -std=c++17 -O3
ifeq ($(debug), 0)
CXXFLAGS += -DNDEBUG
endif

ifeq ($(debug), 1)
CXXFLAGS += -g
endif

ex1: ex1.cu
	nvcc ex1.cu -o ex1.x $(CUDAFLAGS) $(CXXFLAGS) -lcurand

ex2: ex2.cu
	nvcc ex2.cu -o ex2.x $(CUDAFLAGS) $(CXXFLAGS)

ex3: ex3.cu
	nvcc ex3.cu -o ex3.x $(CUDAFLAGS) $(CXXFLAGS)

ex4: ex4.cu
	nvcc ex4.cu -o ex4.x $(CUDAFLAGS) $(CXXFLAGS) -lcurand

clean:
	rm -rf *.x

