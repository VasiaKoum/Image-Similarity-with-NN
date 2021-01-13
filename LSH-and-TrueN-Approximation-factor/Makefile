CXX = g++
all: search
search:
	$(CXX) -o search search.cpp dataset.cpp hash.cpp metrics.cpp lshAlgorithms.cpp
run_search:
	./search -d ../Datasets/train-images-idx3-ubyte -i ../Datasets/reduced_dims_trainset -q ../Datasets/t10k-images-idx3-ubyte -s ../Datasets/reduced_dims_testset -k 3 -L 4 -o out
val_search:
	valgrind --leak-check=yes --track-origins=yes ./search -d ../Datasets/train-images-idx3-ubyte -i ../Datasets/reduced_dims_trainset -q ../Datasets/t10k-images-idx3-ubyte -s ../Datasets/reduced_dims_testset -k 3 -L 4 -o out
clean:
	rm -f search *.o
