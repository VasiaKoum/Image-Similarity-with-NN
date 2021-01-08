CXX = g++
all: lsh cube
lsh:
	$(CXX) -o lsh lsh.cpp dataset.cpp hash.cpp metrics.cpp lshAlgorithms.cpp
cube:
	$(CXX) -o cube cube.cpp dataset.cpp hash.cpp metrics.cpp lshAlgorithms.cpp cubeAlgorithms.cpp projection.cpp
run_lsh:
	./lsh -d ../train-images.idx3-ubyte -R 3000 -q ../t10k-images.idx3-ubyte -k 3 -L 4 -o fileoutlsh -N 2
run_cube:
	./cube -d ../train-images.idx3-ubyte -q ../t10k-images.idx3-ubyte -k 3 -M 10 -probes 2 -o output -Ν 3 -R 3000
val_lsh:
	valgrind --leak-check=yes --track-origins=yes ./lsh -d ../train-images.idx3-ubyte -R 4000 -q ../t10k-images.idx3-ubyte -k 3 -L 1 -o ../../ -N 1
val_cube:
	valgrind --leak-check=yes --track-origins=yes ./cube -d ../train-images.idx3-ubyte -q ../t10k-images.idx3-ubyte -k 3 -M 10 -probes 2 -ο output -Ν 1 -R 1.0
clean:
	rm -f lsh cube *.o
