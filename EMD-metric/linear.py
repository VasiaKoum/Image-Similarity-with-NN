# import the library pulp as p 
from pulp import *
import numpy as np
import pulp as p
import sys
import time
sys.path.insert(1, '../Reduce-Dimensions-Bottleneck-Autoencoder/')
from functions import numpy_from_dataset
from math import sqrt

#python3 linear.py -d ../Datasets/train-images-idx3-ubyte -q ../Datasets/t10k-images-idx3-ubyte -l1 ../Datasets/train-labels-idx1-ubyte
#-l2 ../Datasets/t10k-labels-idx1-ubyte -o results.txt

class cluster:

    def __init__(self, i,j, array, dimSize):
        self.x = i
        self.y = j
        self.dimSize = dimSize
        self.array = array
        self.weight = np.sum(array)
        self.center = (i*dimSize + (dimSize/2), j*dimSize + (dimSize/2))

def distance(a,b):
    return sqrt(abs(b[0] - a[0])**2 + abs(b[1] - a[1])**2)



if ("-d" in sys.argv and "-q" in sys.argv):
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

output = open(output_file, "w")

dims = numarray[2]
clusterDim = 4
step = clusterDim

onetimepass = True

sizeOfCluster = clusterDim*clusterDim
num = dims/clusterDim

correct_emd_results = 0
average_time = 0

for qindex,query in enumerate(qpixels):
    # Consumer

    if qindex == 10:
        break
    query_start_time = time.time()
    consumer = np.reshape(query, (-1, dims))
    consumer_clusters = []
    consumer_firstArrays = np.vsplit(consumer, num)
    for i, array in enumerate(consumer_firstArrays):
        consumer_secondArrays = np.hsplit(array, num)
        ylist = []
        for j, ar in enumerate(consumer_secondArrays):
            ylist.append(cluster(i, j, ar, clusterDim))
        consumer_clusters.append(ylist)

    if onetimepass:
        costs = []
        for x in range(0, len(consumer_clusters)):
            for y in range(0, len(consumer_clusters[x])):
                costsy = []
                for z in range(0, len(consumer_clusters)):
                    for h in range(0, len(consumer_clusters[z])):
                       costsy.append(distance(consumer_clusters[x][y].center,consumer_clusters[z][h].center))
                costs.append(costsy)
        onetimepass = False

    results = []
    for index,image in enumerate(pixels):

        if index == 100:
            break

        #Producer
        producer = np.reshape(image, (-1, dims))
        producer_clusters = []
        producer_firstArrays = np.vsplit(producer, num)
        for i,array in enumerate(producer_firstArrays):
            producer_secondArrays = np.hsplit(array, num)
            ylist = []
            for j,ar in enumerate(producer_secondArrays):
                ylist.append(cluster(i,j,ar, clusterDim))
            producer_clusters.append(ylist)
            # center in cluster = (x*i + mx, y*j + my)
            #weight is np.sum(producer_clusters[0][0])

        Lp_prob = p.LpProblem('Problem', p.LpMinimize)

        xs = [LpVariable("x{}".format(i + 1), lowBound=0) for i in range(int(num)**2)]
        xys = [LpVariable("{}".format(xs[i]) + "y{}".format(j + 1), lowBound=0) for i in range(int(num)**2) for j in range(int(num)**2)]
        xys = np.reshape(xys, (-1, int(num)**2))
        # print(xss)
        #objective function
        statement = 0
        for i in range(int(num)**2):
            for j in range(int(num)**2):
                statement += costs[i][j] * xys[i][j]
        Lp_prob += statement

        #producer constraints
        for i in range(int(num)):
            for j in range(int(num)):
                statement1 = 0
                statement2 = 0
                for z in range(int(num)**2):
                    statement1 += xys[i*int(num) + j][z]
                    statement2 += xys[z][i * int(num) + j]
                # print(statement)
                Lp_prob += statement1 <= producer_clusters[i][j].weight
                Lp_prob += statement2 >= consumer_clusters[i][j].weight

        # # consumer constraints
        # for i in range(int(num)):
        #     for j in range(int(num)):
        #
        #         for z in range(int(num)**2):
        #
        #         # print(statement)
        #

        # print(Lp_prob)
        status = Lp_prob.solve(PULP_CBC_CMD(msg=False))
        # print(p.LpStatus[status])
        # print(p.value(Lp_prob.objective))
        results.append((index, p.value(Lp_prob.objective)))

    results = sorted(results, key=lambda x: x[1])
    success = 0
    for i in range(10):
        if dataset_labels[results[i][0]][0] == queryset_labels[qindex][0]:
            success = success + 1
    correct_emd_results = correct_emd_results + (success/10)
    query_time = time.time() - query_start_time
    average_time = average_time + query_time
    print("query: ", qindex, " nearest neighbour image: ",results[0][0], " with distance: ",results[0][1], dataset_labels[results[0][0]][0]  , queryset_labels[qindex][0], file=output)
    # print("query: ", qindex, " nearest neighbour image: ",results[0][0], " with distance: ",results[0][1])

print("Average Correct Search Results EMD: ", correct_emd_results/10)
print("Average Query Time EMD: ", average_time/10)
output.close()

