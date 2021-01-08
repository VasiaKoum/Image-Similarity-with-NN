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
    unsigned short *image;

    public:
        Neighbor(int, double, unsigned short*);
        int getIndex(){ return index; }
        double getlshDist(){ return lshDist; }
        void printLshNeighbor(int, double,bool, std::ofstream&);
        void printCubeNeighbor(int, double,bool, std::ofstream&);
        double getDist(){return lshDist;};
        unsigned short * getImage(){return image;};
        ~Neighbor();
};

int FindW(int, Dataset*);
void ANNsearch(std::vector<Neighbor>&,int, int, unsigned short*, HashTable**);
void RNGsearch(std::vector<Neighbor>&,int, int, unsigned short*, HashTable**);
void trueDistance(std::vector<double>&, int, unsigned short *, Dataset *, HashTable**);
void trueDistanceWithNeighbors(std::vector<Neighbor>&, unsigned short *, Dataset *, int );



#endif
