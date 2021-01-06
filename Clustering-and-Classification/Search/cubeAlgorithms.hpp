#ifndef CUBEALG
#define CUBEALG

#include <fstream>
#include "dataset.hpp"
#include "projection.hpp"
#include "lshAlgorithms.hpp"

void cubeOutput(ofstream& , int , int , vector<Neighbor>& , vector<Neighbor>& , vector<double>& , vector<double>& ,double , double , double , double );
void hyperCubeSearch(vector<Neighbor>& ,vector<Neighbor>& ,vector<double>& , vector<double>& , double& , double& , double& , double& ,bool ,int ,int , int ,int , unsigned char* , Dataset *, Projection * );
#endif
