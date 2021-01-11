#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include "../LSH-and-TrueN-Approximation-factor/dataset.hpp"

class Centroids{
    double **DParray;
    int *centroids;
    int numpoints;
    int numclusters;
    Dataset *set;

    public:
        Centroids(int, int, Dataset*);
        ~Centroids();
        double** getDParray() { return DParray; }
        int* getCentroids() { return centroids; }
        int getNumClusters() { return numclusters; }
        int getNumPoints() { return numpoints; }
        Dataset* getSet() { return set; }
        double minmaxDist(int, unsigned short*);
        double minDist(std::vector<std::vector<unsigned short>>);
        void Initialize();
};

class Clusters{
    Centroids *Cntrds;
    std::vector<std::vector<unsigned short>> CntrdsVectors;
    std::vector<std::vector<int>> images;
    std::vector<double> snumbers;

    public:
        Clusters(Centroids*);
        void Clustering(char*, std::string);
        void Clustering(std::vector<std::vector<int>>, char*, std::string);
        void Update();
        void Lloyds();
        void Silhouette();
        double ObjectiveFunction();
        void Output(char*, double, std::string, double);
        ~Clusters();
};

void updateDataset(Dataset*, char*, int);
std::vector<std::vector<int>> readClassFile(char*);
