CXX = g++
all: cluster
cluster:
	$(CXX) -o cluster cluster.cpp centroids.cpp ../Search/metrics.cpp ../Search/dataset.cpp ../Search/hash.cpp ../Search/lshAlgorithms.cpp ../Search/projection.cpp ../Search/cubeAlgorithms.cpp
run_cluster:
	./cluster -i ../train-images.idx3-ubyte -c ../cluster.conf -o fileo -m LSH
val_cluster:
	valgrind --leak-check=yes --track-origins=yes ./cluster -i ../train-images.idx3-ubyte -c ../cluster.conf -o fileo -complete 4 -m Classic
clean:
	rm -f cluster *.o
