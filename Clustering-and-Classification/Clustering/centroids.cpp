#include <string.h>
#include <time.h>
#include <random>
#include <cfloat>
#include <algorithm>
#include "../Search/metrics.hpp"
#include "../Search/cubeAlgorithms.hpp"
#include "centroids.hpp"
#define PERCENT_CHANGED_POINTS 99
#define MAX_LOOPS 12
#define RADIUS 8000
#define W 40000

using namespace std;

Centroids::Centroids(int clusters, int points, Dataset *ptrset){
    DParray = new double*[5];
    DParray[0] = new double[points];     //D-Distances
    DParray[1] = new double[points];     //P-Partial sums
    DParray[2] = new double[points];     //Centroid with min dist
    DParray[3] = new double[points];     //D-Distances with second min dist
    DParray[4] = new double[points];     //Centroid with second min dist
    centroids = new int[clusters];       //Indexes of centroids
    numpoints = points;
    numclusters = clusters;
    set = ptrset;
}

Centroids::~Centroids(){
    delete[] DParray[0];
    delete[] DParray[1];
    delete[] DParray[2];
    delete[] DParray[3];
    delete[] DParray[4];
    delete[] DParray;
    delete[] centroids;
}

double Centroids::minmaxDist(int lastcentroid, unsigned char* otherCentroid){
    double manh = 0.0, max = 0.0;
    unsigned char* pointCentroid = NULL;
    bool assignment=false;
    if(otherCentroid == NULL) pointCentroid = set->imageAt(centroids[lastcentroid]);
    else{
        pointCentroid = otherCentroid;
        assignment = true;
    }
    for(int j=0; j<numpoints; j++){
        if(j != centroids[lastcentroid] || assignment){
            manh = manhattan(set->imageAt(j), pointCentroid, set->dimension());
            if(lastcentroid == 0){
                DParray[0][j] = DBL_MAX;
                DParray[2][j] = lastcentroid;
                DParray[3][j] = DBL_MAX;
            }
            if(manh<DParray[0][j]){
                if(DParray[0][j]<DParray[3][j]){
                    DParray[3][j] = DParray[0][j];
                    DParray[4][j] = DParray[2][j];
                }
                DParray[0][j] = manh;
                DParray[2][j] = lastcentroid;
            }
            else{
                if(manh<DParray[3][j]){
                    DParray[3][j] = manh;
                    DParray[4][j] = lastcentroid;
                }
            }
            if(manh>max) max = manh;
        }
        else{
            DParray[0][j] = 0;
            DParray[2][j] = lastcentroid;
        }
    }
    return max;
}

void Centroids::Initialize(){
    random_device randf;
    default_random_engine generator(randf());
    // default_random_engine generator;
    uniform_int_distribution<int> distribution(0,numpoints-1);

    //Choose randomly first centroid
    centroids[0] = distribution(generator);
    cout << "Initialize Centroids with k-Means++ with indexes: " << endl;
    cout << "c0: " << centroids[0] << endl;
    for(int i=1; i<numclusters; i++){
        int sum = 0.0;
        double max = 0.0;
        //Update distances and return max from all distances
        max = minmaxDist(i-1, NULL);
        for(int j=0; j<numpoints; j++){
            sum+=(DParray[0][j]/max)*DParray[0][j];
            DParray[1][j] = sum;
        }
        uniform_real_distribution<> distribution(0,DParray[1][numpoints-1]);
        double find_index = distribution(generator);
        int j = 1, index = 0;
        //Find index from random value
        if(DParray[1][0]>=find_index) index = 0;
        else{
            while(j<numpoints){
                if(DParray[1][j-1]<find_index && DParray[1][j]>=find_index){
                    index = j;
                    break;
                }
                j++;
            }
        }
        centroids[i] = index;
        cout << "c" << i << ": " << centroids[i] << endl;
    }
    //Update for last centroid to use it in Lloyds
    cout << endl;
    minmaxDist(numclusters-1, NULL);
}

Clusters::Clusters(Centroids* argCntrds) : Cntrds(argCntrds){}

void Clusters::Clustering(char* method, char* output, bool complete, int numHashTables, int numHashFunctions, int maxMhypercube, int hypercubeDim, int numprobes){
    clock_t tStart, tClustering, tSilhouette;
    cout << "Begin clustering with " << method << " method for: "  << Cntrds->getNumPoints() << " points." << endl;
    if(strcmp(method, "CLASSIC")==0 || strcmp(method, "classic")==0 || strcmp(method, "Classic")==0){
        tStart = clock();
        Lloyds();
        tClustering = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    }
    else if(strcmp(method, "LSH")==0 || strcmp(method, "lsh")==0 || strcmp(method, "Lsh")==0){
        //Initialize hashtables & hashfunctions
        cout << "Initialize HashTables and HashFunctions..." << endl;
        int points = Cntrds->getNumPoints();
        Dataset *set = Cntrds->getSet();
        int bucketsNumber = floor(points/16);
        HashFunction **hashFamily = new HashFunction*[set->getNumberOfPixels()];
        for(int i=0; i<set->getNumberOfPixels(); i++){
            hashFamily[i] = NULL;
        }
        HashTable **hashTables = new HashTable*[numHashTables];
        for(int i=0; i<numHashTables; i++){
            hashTables[i] = new HashTable(set->getNumberOfPixels(),bucketsNumber,numHashFunctions,W,hashFamily);
            for(int j=0; j<points; j++){
                unsigned int g_hash = (unsigned int)(hashTables[i]->ghash(set->imageAt(j)));
                hashTables[i]->getBucketArray()[g_hash%bucketsNumber]->addImage(j,g_hash,set->imageAt(j));
            }
        }
        tStart = clock();
        LSHReverseAssignment(numHashTables, numHashFunctions, hashTables);
        tClustering = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        for(int i=0; i<set->getNumberOfPixels(); i++){
            if(hashFamily[i]!=NULL){
                delete hashFamily[i];
            }
        }
        delete[] hashFamily;
        for(int i=0; i<numHashTables; i++) delete hashTables[i];
        delete[] hashTables;

        //Find next best neighbor cluster
        FindNextBest();
    }
    else if(strcmp(method, "HYPERCUBE")==0 || strcmp(method, "hypercube")==0 || strcmp(method, "Hypercube")==0){
        //Initialize hypercube structures
        cout << "Initialize HashTables and HashFunctions..." << endl;
        int points = Cntrds->getNumPoints();
        Dataset *set = Cntrds->getSet();
        int bucketsNumber = floor(points/16);
        HashFunction **hashFamily = new HashFunction*[set->getNumberOfPixels()];
        for(int i = 0; i < set->getNumberOfPixels();i++){
            hashFamily[i] = NULL;
        }
        Projection *projection = new Projection(set->getNumberOfPixels(), bucketsNumber, hypercubeDim, W, hashFamily);
        for(int j=0; j<points; j++){
            unsigned int g_hash = (unsigned int)(projection->ghash(set->imageAt(j)));
            projection->getBucketArray()[g_hash%bucketsNumber]->addImage(j,g_hash,set->imageAt(j));
        }
        tStart = clock();
        PROJReverseAssignment(maxMhypercube, numprobes, projection);
        tClustering = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        for(int i=0; i<set->getNumberOfPixels(); i++){
                if(hashFamily[i]!=NULL){
                    delete hashFamily[i];
                }
            }
        delete[] hashFamily;
        delete projection;

        //Find next best neighbor cluster
        FindNextBest();
    }
    else{
        cout << "Wrong method. The available methods are: Classic, LSH, Hypercube." << endl;
        return;
    }
    cout << "Time of clustering: " << (double)tClustering << endl;

    tStart = clock();
    Silhouette();
    tSilhouette = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    Output(method, output, complete, tClustering, tSilhouette);
}

//Update
void Clusters::Update(){
    Dataset *set = Cntrds->getSet();
    int clusters = Cntrds->getNumClusters();
    int *centroids = Cntrds->getCentroids();
    CntrdsVectors.clear();
    vector<unsigned char> newCentroid;
    vector<double> findmedian;
    unsigned char* tmp = NULL;

    for(int i=0; i<clusters; i++){
        cout << "In cluster: " << i << endl;
        int median = ceil(images[i].size()/2);
        newCentroid.clear();
        for (int k=0; k<set->dimension(); k++){
            int allsame = 0;
            int numsame = 0;
            findmedian.clear();
            for(int j=0; j<images[i].size(); j++){
                //For every point in cluster i
                int index = images[i][j];
                tmp = set->imageAt(index);
                findmedian.push_back((double) tmp[k]);
                //Check if all are same number to optimize median
                if(j==0) numsame = (double) tmp[k];
                if((int) tmp[k] == numsame) allsame++;
            }
            //If all are same number then it doesnt need sort and we can pick the median
            if(allsame != images[i].size()){
                sort(findmedian.begin(), findmedian.end());
            }
            newCentroid.push_back(findmedian[median]);
        }
        CntrdsVectors.push_back(newCentroid);
        centroids[i] = i;
    }
}

//Assignment
void Clusters::Lloyds(){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();

    //Initialize Clusters from Centroid initialization
    int *tmp = Cntrds->getCentroids();
    for(int i=0; i<clusters; i++)
        images.push_back(std::vector<int>());
    for(int i=0; i<points; i++)
        images[DParray[2][i]].push_back(i); //Push image to the centroid with min dist DParray[2][i]

    // index = image[from_cluster][what_image]
    int loops = 0;
    int tmpPoint = 0;
    int changedpoints = set->getNumberOfImages();
    int floor_changed_points = points - (points*PERCENT_CHANGED_POINTS/100);
    cout << "Clustering until: changed_points<=" << floor_changed_points << " OR Max_loops=" << MAX_LOOPS << endl;

    //Checking for no-change
    while(changedpoints>floor_changed_points && loops<MAX_LOOPS){
        cout << "IN LOOP: " << loops+1 << "/" << MAX_LOOPS << endl;
        changedpoints = 0;
        //Update Centroids from clusters
        Update();

        // Assignment for all points to the nearest Centroid
        for(int i=1; i<clusters+1; i++)
            Cntrds->minmaxDist(i-1, &CntrdsVectors[i-1][0]);

        //Change points from the old cluster to the new
        for(int i=0; i<clusters; i++){
            for(int j=0; j<images[i].size(); j++){
                int newCluster = DParray[2][images[i][j]];
                if(i!=newCluster){
                    tmpPoint = images[i][j];
                    images[i].erase(images[i].begin()+j);
                    images[newCluster].push_back(tmpPoint);
                    changedpoints++;
                }
            }
        }
        cout << "Changed points: " << changedpoints << "\n\n";
        loops++;
    }
}

void Clusters::FindNextBest(){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();
    double min1, min2, manh;
    int index1, index2;

    for(int i=0; i<points; i++){
        min1 = DBL_MAX, min2 = DBL_MAX;
        for(int j=0; j<clusters; j++){
            manh = manhattan(set->imageAt(i), &CntrdsVectors[j][0], set->dimension());
            if(manh<min1){
                if(min1<min2){
                    min2 = min1;
                    index2 = index1;
                }
                min1 = manh;
                index1 = j;
            }
            else{
                if(manh<min2){
                    min2 = manh;
                    index2 = j;
                }
            }
        }
        if(DParray[0][i]!=min1){
            DParray[3][i] = min1;
            DParray[4][i] = index1;
        }
        else{
            DParray[3][i] = min2;
            DParray[4][i] = index2;
        }
    }
}

void Clusters::AssignReverse(vector<vector<Neighbor>> Neighbors, bool initialize){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    int *centroids = Cntrds->getCentroids();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();
    unsigned char* CentroidVector = NULL;

    if(initialize){
        for(int i=0; i<clusters; i++) CntrdsVectors.push_back(std::vector<unsigned char>());
    }

    //Unassign all points
    for(int i=0; i<points; i++) DParray[0][i] = -1;
    for(int i=0; i<Neighbors.size(); i++){
        for(int j=0; j<Neighbors[i].size(); j++){
            int index = Neighbors[i][j].getIndex();
            //Check if point is unassigned or it has been already assigned
            if(DParray[0][index]<0){
                DParray[0][index] = Neighbors[i][j].getlshDist();
                DParray[2][index] = i;
            }
            else{
                //If is already assigned then find the nearest Centroid
                if(initialize) CentroidVector = set->imageAt(centroids[i]);
                else CentroidVector = &CntrdsVectors[i][0];
                double manh = manhattan(set->imageAt(index), CentroidVector, set->dimension());
                if(manh<DParray[0][index]){
                    DParray[0][index] = manh;
                    DParray[2][index] = i;
                }
            }

        }
    }
    double min = 0, manh = 0;
    //For unassigned points
    for(int i=0; i<points; i++){
        if(DParray[0][i]<0){
            min = DBL_MAX;
            for(int j=0; j<clusters; j++){
                if(initialize) CentroidVector = set->imageAt(centroids[j]);
                else CentroidVector = &CntrdsVectors[j][0];
                manh = manhattan(set->imageAt(i), CentroidVector, set->dimension());
                if(manh<min){
                    DParray[0][i] = manh;
                    DParray[2][i] = j;
                    min = manh;
                }
            }
        }
    }
}

void Clusters::LSHReverseAssignment(int numHashTables, int numHashFunctions, HashTable **hashTables){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    int *centroids = Cntrds->getCentroids();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();
    int radius = RADIUS;

    //Initialize images from centroids
    vector<vector<Neighbor>> RNGneighbors;
    for(int i=0; i<clusters; i++){
        RNGneighbors.push_back(std::vector<Neighbor>());
        RNGsearch(RNGneighbors[i], numHashTables, radius, set->imageAt(centroids[i]), hashTables);
    }
    AssignReverse(RNGneighbors, true);
    //Initialize clusters from DParray structure
    int *tmp = Cntrds->getCentroids();
    for(int i=0; i<clusters; i++)
        images.push_back(std::vector<int>());
    for(int i=0; i<points; i++)
        images[DParray[2][i]].push_back(i); //Push image to the centroid with min dist DParray[2][i]

    int loops = 0;
    int tmpPoint = 0;
    int changedpoints = set->getNumberOfImages();
    int floor_changed_points = points - (points*PERCENT_CHANGED_POINTS/100);
    cout << "Clustering until: changed_points<=" << floor_changed_points << " OR Max_loops=" << MAX_LOOPS << endl;

    //Checking for no-change
    while(changedpoints>floor_changed_points && loops<MAX_LOOPS){
        cout << "IN LOOP: " << loops+1 << "/" << MAX_LOOPS << endl;
        changedpoints = 0;
        //Clear neighbors
        for(int i=0; i<RNGneighbors.size(); i++) RNGneighbors[i].clear();

        //Update Centroids from clusters
        Update();

        // Assignment
        for(int i=0; i<clusters; i++)
            RNGsearch(RNGneighbors[i], numHashTables, radius, &CntrdsVectors[i][0], hashTables);
        AssignReverse(RNGneighbors, false);

        //Change points from the old cluster to the new
        for(int i=0; i<clusters; i++){
            for(int j=0; j<images[i].size(); j++){
                int newCluster = DParray[2][images[i][j]];
                if(i!=newCluster){
                    tmpPoint = images[i][j];
                    images[i].erase(images[i].begin()+j);
                    images[newCluster].push_back(tmpPoint);
                    changedpoints++;
                }
            }
        }
        cout << "Changed points: " << changedpoints << "\n\n";
        radius = radius*2;
        loops++;
    }
}

void Clusters::PROJReverseAssignment(int maxMhypercube, int numprobes, Projection* projection){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    int *centroids = Cntrds->getCentroids();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();
    vector<Neighbor> ANNneighbors;
    vector<double> ANNtrueDist, RNGtrueDist;
    double cubeAnnTime=0, cubeRngTime=0, trueAnnTime=0, trueRngTime=0;
    int radius = RADIUS;

    //Initialize images from centroids
    vector<vector<Neighbor>> RNGneighbors;
    for(int i=0; i<clusters; i++){
        RNGneighbors.push_back(std::vector<Neighbor>());
        hyperCubeSearch(ANNneighbors, RNGneighbors[i], ANNtrueDist, RNGtrueDist, cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime, false, radius, 0, numprobes, i, set->imageAt(centroids[i]), set, projection);
    }
    AssignReverse(RNGneighbors, true);
    //Initialize clusters from DParray structure
    int *tmp = Cntrds->getCentroids();
    for(int i=0; i<clusters; i++)
        images.push_back(std::vector<int>());
    for(int i=0; i<points; i++)
        images[DParray[2][i]].push_back(i); //Push image to the centroid with min dist DParray[2][i]

    int loops = 0;
    int tmpPoint = 0;
    int changedpoints = set->getNumberOfImages();
    int floor_changed_points = points - (points*PERCENT_CHANGED_POINTS/100);
    cout << "Clustering until: changed_points<=" << floor_changed_points << " OR Max_loops=" << MAX_LOOPS << endl;

    //Checking for no-change
    while(changedpoints>floor_changed_points && loops<MAX_LOOPS){
        cout << "IN LOOP: " << loops+1 << "/" << MAX_LOOPS << endl;
        changedpoints = 0;
        //Clear neighbors
        for(int i=0; i<RNGneighbors.size(); i++) RNGneighbors[i].clear();

        //Update Centroids from clusters
        Update();

        // Assignment
        for(int i=0; i<clusters; i++)
            hyperCubeSearch(ANNneighbors, RNGneighbors[i], ANNtrueDist, RNGtrueDist, cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime, false, radius, 0, numprobes, i, &CntrdsVectors[i][0], set, projection);
        AssignReverse(RNGneighbors, false);

        //Change points from the old cluster to the new
        for(int i=0; i<clusters; i++){
            for(int j=0; j<images[i].size(); j++){
                int newCluster = DParray[2][images[i][j]];
                if(i!=newCluster){
                    tmpPoint = images[i][j];
                    images[i].erase(images[i].begin()+j);
                    images[newCluster].push_back(tmpPoint);
                    changedpoints++;
                }
            }
        }
        cout << "Changed points: " << changedpoints << "\n\n";
        radius = radius*2;
        loops++;
    }
}

void Clusters::Silhouette(){
    int points = Cntrds->getNumPoints();
    int clusters = Cntrds->getNumClusters();
    double **DParray = Cntrds->getDParray();
    Dataset *set = Cntrds->getSet();

    double a;
    double b;
    int index1 = 0;
    int index2 = 0;
    double manh = 0, sum_a = 0, sum_b = 0, totalsum = 0, clusterssum = 0, max = 0, result = 0;

    cout << "\nBegin silhouette with for: "  << points << " points." << endl;
    for(int i=0; i<images.size(); i++){
        cout << "CLUSTER " << i << "/" << images.size() << endl;
        clusterssum = 0;
        for(int j=0; j<images[i].size(); j++){
            sum_a = 0;
            sum_b = 0;
            // a(i) -> Find average dist from all points in same cluster
            for(int k=0; k<images[i].size(); k++){
                if(j!=k){
                    index1 = images[i][j];
                    index2 = images[i][k];
                    sum_a+= manhattan(set->imageAt(index1), set->imageAt(index2), set->dimension());
                }
            }
            a = (sum_a/(images[i].size()-1));
            int nextBestCluster = DParray[4][j];
            // b(i) -> Find average dist from all points in next best cluster
            for(int k=0; k<images[nextBestCluster].size(); k++){
                index1 = images[i][j];
                index2 = images[nextBestCluster][k];
                sum_b+=manhattan(set->imageAt(index1), set->imageAt(index2), set->dimension());
            }
            b = (sum_b/images[nextBestCluster].size());

            //Calculate s(i) numbers:
            max = a;
            if(a<b) max = b;
            if(max!=0){
                result = (b-a)/max;
            }
            else result = 0;
            totalsum+=result;
            clusterssum+=result;
        }
        result = clusterssum/images[i].size();
        snumbers.push_back(result);
        cout << "For " << i << " Silhouette is: " << clusterssum/images[i].size() << endl;
    }
    result = totalsum/points;
    snumbers.push_back(result);
    cout << "Total Silhouette is: " << result;
}

void Clusters::Output(char* method, char *output, bool complete, double tClustering, double tSilhouette){
    ofstream outfile((char*)output);
    if (outfile.is_open()){
        outfile << "Algorithm: "<< method << "\n";
        outfile << "Time Clustering: "<< (double)tClustering << " sec" << "\n";
        outfile << "Time Silhouette: "<< (double)tSilhouette << " sec"<< "\n";
        for(int i=0; i<CntrdsVectors.size(); i++){
            outfile << "CLUSTER-" << i+1 << " {size: " << images[i].size() << ", centroid: ";
            for(int j=0; j<CntrdsVectors[i].size(); j++){
                outfile << (int)CntrdsVectors[i][j] << " ";
            }
            outfile << "}\n\n";
        }

        //Silhouette
        for(int i=0; i<snumbers.size(); i++){
            if(i!=snumbers.size()-1){
                outfile << "s" << i << "=" << snumbers[i] << ", ";
            }
            else outfile << "stotal = " << snumbers[i];
        }
        outfile << "\n\n";

        if(complete){
            for(int i=0; i<CntrdsVectors.size(); i++){
                outfile << "CLUSTER-" << i+1 << " { ";
                for(int j=0; j<CntrdsVectors[i].size(); j++){
                    outfile << (int)CntrdsVectors[i][j] << " ";
                }
                outfile << ", ";
                for(int j=0; j<images[i].size(); j++){
                    outfile << images[i][j];
                    if(j!=images[i].size()-1) outfile << ", ";
                }
                outfile << "}\n\n";
            }
        }

        outfile.close();
    }
    else cout << "Unable to open outout file" << endl;
}

Clusters::~Clusters(){

}
