#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cfloat>
#include <math.h>
#include <fstream>
#include "dataset.hpp"
#include "metrics.hpp"
#include "lshAlgorithms.hpp"
#include "hash.hpp"

using namespace std;

int FindW(int samples, Dataset *set){
    double sum=0.0, min, tmp;
    int d = set->dimension(), img = set->getNumberOfImages();
    for(int i=0; i<samples; i++){
        min=0.0;
        for(int j=0; j<img; j++){
            if(i!=j){
                if(min!=0.0){
                    tmp = manhattan(set->imageAt(i), set->imageAt(j), d);
                    if(tmp<min) min = tmp;
                }
                else min = manhattan(set->imageAt(i), set->imageAt(j), d);
            }
        }
        sum+=min;
    }
    return sum/samples;
}

void trueDistance(vector<double>& trueDist, int R, unsigned char *q, Dataset *trainSet, HashTable** hashTables){
    double min, manh=0.0;
    if(R>0){
        for(int i=0; i<trainSet->getNumberOfImages(); i++){
        manh = manhattan(q, trainSet->imageAt(i),  hashTables[0]->getvectorsDim());
            if(manh<R) {
                trueDist.push_back(manh);
            }
        }
    }
    else{
        min = DBL_MAX;
        for(int i=0; i<trainSet->getNumberOfImages(); i++){
            manh = manhattan(q, trainSet->imageAt(i),  hashTables[0]->getvectorsDim());
            trueDist.push_back(manh);
        }
    }

    return;

}

void ANNsearch(vector<Neighbor>& neighbors, int L, int k, unsigned char *q, HashTable** hashTables){
    unsigned int g_hash = 0;
    double min, manh=0.0;
    for(int i=0; i<L; i++){
        g_hash = hashTables[i]->ghash(q);
        min = DBL_MAX;
        vector<imageInfo>* images = hashTables[i] ->getBucketArray()[g_hash%(hashTables[i] ->gethashTableSize())] -> getImageList();
        for (vector<imageInfo>::iterator it = images->begin() ; it != images->end(); ++it){
            if((*it).ghash == g_hash){
                manh = manhattan((*it).image, q, hashTables[i]->getvectorsDim());
                if(manh < min){
                    min = manh;
                    neighbors.push_back(Neighbor((*it).index, min, (*it).image));
                }
                // if(j>15*L) break;
                // j++;
            }
        }
    }

    return;
}

void RNGsearch(vector<Neighbor>& neighbors, int L, int R, unsigned char* q, HashTable** hashTables){
    unsigned int g_hash = 0;
    double manh=0.0;
    for(int i=0; i<L; i++){
        g_hash = hashTables[i]->ghash(q);
        vector<imageInfo>* images = hashTables[i] ->getBucketArray()[g_hash%(hashTables[i] ->gethashTableSize())] -> getImageList();
        for (vector<imageInfo>::iterator it = images->begin() ; it != images->end(); ++it){
            if((*it).ghash == g_hash){
                manh = manhattan((*it).image, q, hashTables[i]->getvectorsDim());
                if(manh < R){
                    neighbors.push_back(Neighbor((*it).index, manh, (*it).image));
                }
                // if(j>15*L) break;
                // j++;
            }
        }
    }
    return;
}



Neighbor::Neighbor(int i, double l, unsigned char* img): index(i), lshDist(l){
    image = img;
};

void Neighbor::printLshNeighbor(int i, double trueDist,bool range, ofstream& outputf){
    if(range){
        outputf << "neighbor-" << i << ": " << index << endl;
        outputf << "distanceLSH: " << lshDist << endl;
    }else{
        outputf << "Nearest neighbor-" << i << ": " << index << endl;
        outputf << "distanceLSH: " << lshDist << endl;
        outputf << "distanceTrue: " << trueDist << endl;
    }
};

void Neighbor::printCubeNeighbor(int i, double trueDist,bool range, ofstream& outputf){
    if(range){
        outputf << "neighbor-" << i << ": " << index << endl;
        outputf << "distanceCube: " << lshDist << endl;
    }else{
        outputf << "Nearest neighbor-" << i << ": " << index << endl;
        outputf << "distanceCube: " << lshDist << endl;
        outputf << "distanceTrue: " << trueDist << endl;
    }
};

Neighbor::~Neighbor(){};
