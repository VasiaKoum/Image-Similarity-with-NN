#include "projection.hpp"
#include <random>

using namespace std;

Projection::Projection(int vectordim,int numberOfBuckets, int projdim, int W, HashFunction** hashFamily): HashTable(vectordim,numberOfBuckets,projdim,W,hashFamily){
    
    // hashFunctions = new HashFunction*[this->bucketDim];
    // for(int i=0; i<this->bucketDim; i++){
    //     hashFunctions[i] = new HashFunction(this->vectorsDim, W, this->bucketDim);
    // }
    hStorage = new map<int, int>[projdim];
}

Projection::~Projection(){
    delete[] hStorage;
}

unsigned int Projection::ghash(unsigned short * image){
    random_device randf;
    static default_random_engine binary_generator(randf());
    uniform_int_distribution<int> binary_distribution(0,1);
    
    int result = 0;
    int bit = 0;
    for(int i=0; i<this->numberOfHashFuncs; i++){
        int hash = 0;
        map<int, int>::iterator itr;
        hash = hashFunctions[i]->hash(image);
        itr = hStorage[i].find(hash);
        if (itr != hStorage[i].end()){
            bit = itr->second;
        }else{
            bit = binary_distribution(binary_generator);
            hStorage[i].insert(pair<int, int>(hash, bit));
        }
        if(i != 0){
            result = result<<1;
            result = result | bit;
        }else{
            result = result | bit;
        }    
    }
    return result;
}