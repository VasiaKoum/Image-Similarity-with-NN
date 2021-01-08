#include "../Search/dataset.hpp"
#include "../Search/lshAlgorithms.hpp"
#include "../Search/projection.hpp"
#include "../Search/hash.hpp"

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
        void Clustering(char*, char*, bool, int, int, int, int, int);
        void AssignReverse(std::vector<std::vector<Neighbor>>, bool);
        void LSHReverseAssignment(int, int, HashTable**);
        void PROJReverseAssignment(int, int, Projection*);
        void Update();
        void Lloyds();
        void FindNextBest();
        void Silhouette();
        void Output(char*, char*, bool, double, double);
        ~Clusters();
};
