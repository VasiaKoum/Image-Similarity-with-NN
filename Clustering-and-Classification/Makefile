CXX = g++
all: cluster
cluster:
	$(CXX) -o cluster ./Clustering/cluster.cpp ./Clustering/centroids.cpp ./Search/metrics.cpp ./Search/dataset.cpp ./Search/hash.cpp ./Search/lshAlgorithms.cpp ./Search/projection.cpp ./Search/cubeAlgorithms.cpp
	# $(CXX) -o cluster ./Clustering/cluster.cpp ./Clustering/centroids.cpp ../LSH-and-TrueN-Approximation-factor/metrics.cpp ../LSH-and-TrueN-Approximation-factor/dataset.cpp ../LSH-and-TrueN-Approximation-factor/hash.cpp ../LSH-and-TrueN-Approximation-factor/lshAlgorithms.cpp ../LSH-and-TrueN-Approximation-factor/projection.cpp ../LSH-and-TrueN-Approximation-factor/cubeAlgorithms.cpp
run_cluster:
	./cluster -d ../Datasets/reduced_dims_trainset -i ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o fileo
val_cluster:
	valgrind --leak-check=yes --track-origins=yes ./cluster -d ../Datasets/reduced_dims_trainset -i ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o fileo
clean:
	rm -f cluster *.o
