# import the library pulp as p 
from pulp import *
import numpy as np
import pulp as p
import sys
sys.path.insert(1, '../Reduce-Dimensions-Bottleneck-Autoencoder/')
from functions import numpy_from_dataset
from math import sqrt

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

output = open("results.txt", "w")

if ("-d" in sys.argv and "-q" in sys.argv):
    dataset = sys.argv[sys.argv.index("-d")+1]
    queryset = sys.argv[sys.argv.index("-q")+1]
    # output_data = sys.argv[sys.argv.index("-od")+1]
    # output_query = sys.argv[sys.argv.index("-oq")+1]
else:
    sys.exit("Wrong or missing parameter. Please execute with: -d dataset -q queryset -od output_data -oq output_query")
pixels, numarray = numpy_from_dataset(dataset, 4, False)
qpixels, qnumarray = numpy_from_dataset(queryset, 4, False)

dims = 28
clusterDim = 4
step = clusterDim

onetimepass = True

sizeOfCluster = clusterDim*clusterDim
num = dims/clusterDim

for qindex,query in enumerate(qpixels):
    # Consumer

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
        break
    sorted(results, key=lambda x: x[1])
    print("query: ", qindex, " nearest neighbour image: ",results[0][0], " with distance: ",results[0][1], file=output)
    print("query: ", qindex, " nearest neighbour image: ",results[0][0], " with distance: ",results[0][1])
    break
output.close()

