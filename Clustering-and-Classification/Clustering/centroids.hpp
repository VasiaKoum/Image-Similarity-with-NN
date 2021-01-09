#include <iostream>
#include <fstream>
#include <vector>
#include "../Search/dataset.hpp"
// #include "../../LSH-and-TrueN-Approximation-factor/lshAlgorithms.hpp"

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
        double minmaxDist(int, unsigned char*);
        void Initialize();
};

class Clusters{
    Centroids *Cntrds;
    std::vector<std::vector<unsigned char>> CntrdsVectors;
    std::vector<std::vector<int>> images;
    std::vector<double> snumbers;

    public:
        Clusters(Centroids*);
        void Clustering(char*);
        void Update();
        void Lloyds();
        void FindNextBest();
        void Silhouette();
        void Output(char*, double, double);
        ~Clusters();
};
