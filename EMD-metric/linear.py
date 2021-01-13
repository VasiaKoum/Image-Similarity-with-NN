from pulp import *
import numpy as np
import pulp as p
import sys
import time
from itertools import zip_longest
# sys.path.insert(1, '../Reduce-Dimensions-Bottleneck-Autoencoder/')
# from functions import numpy_from_dataset
from math import sqrt

#python3 linear.py -d ../Datasets/train-images-idx3-ubyte -q ../Datasets/t10k-images-idx3-ubyte -l1 ../Datasets/train-labels-idx1-ubyte
#-l2 ../Datasets/t10k-labels-idx1-ubyte -o results.txt

def numpy_from_dataset(inputpath, numbers, per_2_bytes):
    pixels = []
    numarray = []
    with open(inputpath, "rb") as file:
        for x in range(numbers):
            numarray.append(int.from_bytes(file.read(4), byteorder='big'))
        # 2d numpy array for images->pixels
        if numbers == 4:
            if per_2_bytes:
                data = file.read(2)
                while data:
                    pixels.append(int.from_bytes(data, byteorder='big'))
                    data = file.read(2)
                pixels = np.array(list(bytes_group(numarray[3], pixels, fillvalue=0)))
            else:
                pixels = np.array(list(bytes_group(numarray[2]*numarray[3], file.read(), fillvalue=0)))
        elif numbers == 2:
            pixels = np.array(list(bytes_group(1, file.read(), fillvalue=0)))
    return pixels, numarray

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

class cluster:

    def __init__(self, i,j, array, dimSize, sum_pixels):
        self.x = i
        self.y = j
        self.dimSize = dimSize
        self.array = array
        self.weight = int((np.sum(array)/sum_pixels)*100)
        self.center = (i*dimSize + (dimSize/2), j*dimSize + (dimSize/2))


def distance(a,b):
    # return np.linalg.norm(np.array([a[0], a[1]] - np.array([b[0], b[1]])))
    return sqrt(abs(b[0] - a[0])**2 + abs(b[1] - a[1])**2)


if "-d" in sys.argv and "-q" in sys.argv:
    dataset = sys.argv[sys.argv.index("-d")+1]
    queryset = sys.argv[sys.argv.index("-q")+1]
    datasetLabels = sys.argv[sys.argv.index("-l1") + 1]
    querysetLabels = sys.argv[sys.argv.index("-l2") + 1]
    output_file = sys.argv[sys.argv.index("-o")+1]
else:
    sys.exit("Wrong or missing parameter. Please execute with: -d dataset -q queryset -od output_data -oq output_query")

pixels, numarray = numpy_from_dataset(dataset, 4, False)
qpixels, qnumarray = numpy_from_dataset(queryset, 4, False)
dataset_labels, numarray_labels = numpy_from_dataset(datasetLabels, 2, False)
queryset_labels, qnumarray_labels = numpy_from_dataset(querysetLabels, 2, False)

clusterDim = 7 #cluster dimension n (nxn)

output = open(output_file, "a")
dims = numarray[2]
step = clusterDim

onetimepass = True

sizeOfCluster = clusterDim*clusterDim
num = dims/clusterDim

correct_emd_results = 0
average_time = 0

for qindex,query in enumerate(qpixels):

    # if qindex < 10:
    #     continue
    if qindex == 10:
        break

    # Consumer
    sum_consumer = np.sum(query)
    query_start_time = time.time()
    consumer = np.reshape(query, (-1, dims))
    consumer_clusters = []
    sum_consumer_100 = 0

    consumer_firstArrays = np.vsplit(consumer, num)
    for i, array in enumerate(consumer_firstArrays):
        consumer_secondArrays = np.hsplit(array, num)
        ylist = []
        for j, ar in enumerate(consumer_secondArrays):
            new_cluster = cluster(i, j, ar, clusterDim,sum_consumer)
            sum_consumer_100 = sum_consumer_100 + new_cluster.weight
            ylist.append(new_cluster)
        consumer_clusters.append(ylist)
    dif = sum_consumer_100 - 100
    it = 0

    while consumer_clusters[it][0].weight - dif < 0 and it < clusterDim:
        it += 1
        if consumer_clusters[it][0].weight -1 > 0:
            consumer_clusters[it][0].weight -= 1
            dif -=1
    consumer_clusters[it][0].weight -= dif

    if onetimepass:
        costs = []
        for x in range(0, len(consumer_clusters)):
            for y in range(0, len(consumer_clusters[x])):
                costsy = []
                for z in range(0, len(consumer_clusters)):
                    for h in range(0, len(consumer_clusters[z])):
                        costsy.append(distance(consumer_clusters[x][y].center,consumer_clusters[z][h].center))
                # print(len(costsy), costsy)
                costs.append(costsy)
        onetimepass = False

    results = []
    for index,image in enumerate(pixels):

        if index == 100:
            break

        sum_producer = np.sum(image)
        sum_producer_100 = 0
        #Producer
        producer = np.reshape(image, (-1, dims))
        producer_clusters = []
        producer_firstArrays = np.vsplit(producer, num)
        for i,array in enumerate(producer_firstArrays):
            producer_secondArrays = np.hsplit(array, num)
            ylist = []
            for j,ar in enumerate(producer_secondArrays):
                new_cluster = cluster(i, j, ar, clusterDim, sum_producer)
                sum_producer_100 = sum_producer_100 + new_cluster.weight
                ylist.append(new_cluster)
            producer_clusters.append(ylist)
        dif = sum_producer_100 - 100
        it = 0
        while producer_clusters[it][0].weight - dif < 0 and it < clusterDim:
            it += 1
            if producer_clusters[it][0].weight - 1 > 0:
                producer_clusters[it][0].weight -= 1
                dif -= 1
        producer_clusters[it][0].weight -= dif

        Lp_prob = p.LpProblem('Problem', p.LpMinimize)

        xs = [LpVariable("x{}".format(i + 1), lowBound=0) for i in range(int(num)**2)]
        xys = [LpVariable("{}".format(xs[i]) + "y{}".format(j + 1), lowBound=0) for i in range(int(num)**2) for j in range(int(num)**2)]
        xys = np.reshape(xys, (-1, int(num)**2))

        #objective function
        statement = 0
        for i in range(int(num)**2):
            for j in range(int(num)**2):
                statement += costs[i][j] * xys[i][j]
        Lp_prob += statement

        #constraints
        for i in range(int(num)):
            for j in range(int(num)):
                statement1 = 0
                statement2 = 0
                for z in range(int(num)**2):
                    statement1 += xys[i*int(num) + j][z]
                    statement2 += xys[z][i * int(num) + j]

                Lp_prob += statement1 <= producer_clusters[i][j].weight
                Lp_prob += statement2 >= consumer_clusters[i][j].weight

        status = Lp_prob.solve(PULP_CBC_CMD(msg=False))
        results.append((index, p.value(Lp_prob.objective.value())))

    results = sorted(results, key=lambda x: x[1])
    success = 0
    for i in range(10):
        if dataset_labels[results[i][0]][0] == queryset_labels[qindex][0]:
            success = success + 1

    correct_emd_results = correct_emd_results + (success/10)
    query_time = time.time() - query_start_time
    average_time = average_time + query_time
    # print("query: ", qindex, " nearest neighbour image: ",results[0][0], " with distance: ",results[0][1], dataset_labels[results[0][0]][0]  , queryset_labels[qindex][0], file=output)

print("Average Correct Search Results EMD: ", correct_emd_results/10)
print("Average Query Time EMD: ", average_time/10)
print("Average Correct Search Results EMD: ", correct_emd_results/10, file=output)
print("Average Query Time EMD: ", average_time/10, file=output)
output.close()

