CXX = g++
all: cluster
cluster:
	$(CXX) -o cluster ./Clustering/cluster.cpp ./Clustering/centroids.cpp ./Search/metrics.cpp ./Search/dataset.cpp
	# $(CXX) -o cluster ./Clustering/cluster.cpp ./Clustering/centroids.cpp ../LSH-and-TrueN-Approximation-factor/metrics.cpp ../LSH-and-TrueN-Approximation-factor/dataset.cpp
run_cluster:
	./cluster -d ../Datasets/reduced_dims_trainset -i ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o fileo
val_cluster:
	valgrind --leak-check=yes --track-origins=yes ./cluster -d ../Datasets/reduced_dims_trainset -i ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o fileo
clean:
	rm -f cluster *.o
