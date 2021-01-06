#ifndef ALGO
#define ALGO

#include <vector>
#include "hash.hpp"
#include "dataset.hpp"
#include <fstream>

class Neighbor{
    // int indexq;
    int index;
    double lshDist;
    unsigned char *image;

    public:
        Neighbor(int, double, unsigned char*);
        int getIndex(){ return index; }
        double getlshDist(){ return lshDist; }
        void printLshNeighbor(int, double,bool, std::ofstream&);
        void printCubeNeighbor(int, double,bool, std::ofstream&);
        double getDist(){return lshDist;};
        ~Neighbor();
};

int FindW(int, Dataset*);
void ANNsearch(std::vector<Neighbor>&,int, int, unsigned char*, HashTable**);
void RNGsearch(std::vector<Neighbor>&,int, int, unsigned char*, HashTable**);
void trueDistance(std::vector<double>&, int, unsigned char *, Dataset *, HashTable**);



#endif
