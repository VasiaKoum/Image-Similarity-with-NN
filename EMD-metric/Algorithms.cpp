#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cfloat>
#include <math.h>
#include <fstream>
#include "dataset.hpp"
#include "metrics.hpp"
#include "Algorithms.hpp"

using namespace std;

bool compareNeighbor(Neighbor& i,Neighbor& j) { return (i.getDist() > j.getDist()); }

void trueDistanceWithNeighbors(vector<Neighbor>& neighbors, unsigned short *q, Dataset *trainSet, int vectorsDim){
    double min, manh=0.0;
        min = DBL_MAX;
        for(int i=0; i<trainSet->getNumberOfImages(); i++){
            manh = manhattan(q, trainSet->imageAt(i), vectorsDim);
            neighbors.push_back(Neighbor(i, manh, trainSet->imageAt(i)));
        }    
    return;

}

Neighbor::Neighbor(int i, double l, unsigned short* img): index(i), lshDist(l){
    image = img;
};

Neighbor::~Neighbor(){};
