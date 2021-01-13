CXX = g++
all: cluster
cluster:
	$(CXX) -o cluster cluster.cpp centroids.cpp ../LSH-and-TrueN-Approximation-factor/metrics.cpp ../LSH-and-TrueN-Approximation-factor/dataset.cpp
run_cluster:
	# ./cluster -i ../Datasets/reduced_dims_trainset -d ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o fileo
	./cluster -i ../Datasets/reduced_dims_testset -d ../Datasets/t10k-images-idx3-ubyte -n classes_L4_FS32_E200_B64 -c cluster.conf -o cluster_output
val_cluster:
	valgrind --leak-check=yes --track-origins=yes ./cluster -i ../Datasets/reduced_dims_trainset -d ../Datasets/train-images-idx3-ubyte -n classes -c cluster.conf -o cluster_output
clean:
	rm -f cluster *.o
