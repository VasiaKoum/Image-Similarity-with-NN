#ifndef ALGO
#define ALGO

#include <vector>
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
        double getDist(){return lshDist;};
        unsigned short * getImage(){return image;};
        ~Neighbor();
};
bool compareNeighbor(Neighbor&,Neighbor&);
void trueDistanceWithNeighbors(std::vector<Neighbor>&, unsigned short *, Dataset *, int );

#endif
