CXX = g++
all: search
search:
	$(CXX) -o search search.cpp dataset.cpp metrics.cpp Algorithms.cpp
run_search:
	./search -d ../Datasets/train-images-idx3-ubyte -q ../Datasets/t10k-images-idx3-ubyte -l1 ../Datasets/train-labels-idx1-ubyte -l2 ../Datasets/t10k-labels-idx1-ubyte -o out
val_search:
	valgrind --leak-check=yes --track-origins=yes ./search -d ../Datasets/train-images-idx3-ubyte -q ../Datasets/t10k-images-idx3-ubyte -l1 ../Datasets/train-labels-idx1-ubyte -l2 ../Datasets/t10k-labels-idx1-ubyte -o out
clean:
	rm -f search *.o
