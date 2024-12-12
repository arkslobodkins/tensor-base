all : ex1 ex2 ex3 ex4 ex5

debug = 1

CXXFLAGS = --std=c++17 -O3 -arch=sm_89
ifeq ($(debug), 0)
CXXFLAGS += -DNDEBUG
endif

ifeq ($(debug), 1)
CXXFLAGS += -g
endif

IFLAGS =-I../src/

ex1: ex1.cu
	nvcc ex1.cu -o ex1.x $(CXXFLAGS) $(IFLAGS) -lcurand

ex2: ex2.cu
	nvcc ex2.cu -o ex2.x $(CXXFLAGS) $(IFLAGS)

ex3: ex3.cu
	nvcc ex3.cu -o ex3.x $(CXXFLAGS) $(IFLAGS)

ex4: ex4.cu
	nvcc ex4.cu -o ex4.x $(CXXFLAGS) $(IFLAGS) -lcurand

ex5: ex5.cu
	nvcc ex5.cu -o ex5.x $(CXXFLAGS) $(IFLAGS) -lcurand

clean:
	rm -rf *.x

