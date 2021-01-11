#include <time.h>
#include <random>
#include <cfloat>
#include <algorithm>
#include "../LSH-and-TrueN-Approximation-factor/metrics.hpp"
#include "centroids.hpp"
#define PERCENT_CHANGED_POINTS 99
#define MAX_LOOPS 1
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

double Centroids::minDist(vector<vector<unsigned short>> CntrdsVectors){
    double manh, sum=0;
    for(int i=0; i<numpoints; i++){
        DParray[0][i] = DBL_MAX; DParray[3][i] = DBL_MAX;
        for(int j=0; j<CntrdsVectors.size(); j++){
            manh = manhattan(set->imageAt(i), &CntrdsVectors[j][0], set->dimension());
            if(manh<DParray[0][i]){
                if(DParray[0][i]<DParray[3][i]){
                    DParray[3][i] = DParray[0][i];
                    DParray[4][i] = DParray[2][i];
                }
                DParray[0][i] = manh;
                DParray[2][i] = j;
            }
            else{
                if(manh<DParray[3][i]){
                    DParray[3][i] = manh;
                    DParray[4][i] = j;
                }
            }

        }
        sum+=DParray[0][i];
    }
    return sum;
}

double Centroids::minmaxDist(int lastcentroid, unsigned short* otherCentroid){
    double manh = 0.0, max = 0.0;
    unsigned short* pointCentroid = NULL;
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

void Clusters::Clustering(char* output, string typecluster){
    clock_t tStart, tClustering, tSilhouette;
    cout << "Begin clustering with Classic method for: " << Cntrds->getNumPoints() << " points." << endl;
    tStart = clock();
    Lloyds();
    tClustering = (double)(clock() - tStart)/CLOCKS_PER_SEC;
    cout << "Time of clustering: " << (double)tClustering << endl;
    // Silhouette();
    Output(output, tClustering, typecluster, ObjectiveFunction());
}

//Update
void Clusters::Update(){
    Dataset *set = Cntrds->getSet();
    int clusters = Cntrds->getNumClusters();
    int *centroids = Cntrds->getCentroids();
    CntrdsVectors.clear();
    vector<unsigned short> newCentroid;
    vector<double> findmedian;
    unsigned short* tmp = NULL;

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

void Clusters::Clustering(vector<vector<int>> argimages, char* output, string typecluster){
    double **DParray = Cntrds->getDParray();
    int numclusters = Cntrds->getNumClusters();
    images = argimages;
    Update();
    // Silhouette();
    Output(output, 0, typecluster, Cntrds->minDist(CntrdsVectors));
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

    cout << "\nBegin silhouette for: "  << points << " points." << endl;
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

double Clusters::ObjectiveFunction(){
    int points = Cntrds->getNumPoints();
    double **DParray = Cntrds->getDParray(), sum=0;
    for(int i=0; i<points; i++) sum+=DParray[0][i];
    return sum;
}

void Clusters::Output(char *output, double tClustering, string typecluster, double objectiveSum){
    ofstream outfile((char*)output, ios::app);
    if (outfile.is_open()){
        outfile << typecluster << endl;
        if(typecluster.compare("CLASSES AS CLUSTERS") != 0){
            for(int i=0; i<CntrdsVectors.size(); i++){
                outfile << "CLUSTER-" << i+1 << " {size: " << images[i].size() << ", centroid: ";
                for(int j=0; j<CntrdsVectors[i].size(); j++){
                    outfile << (int)CntrdsVectors[i][j] << " ";
                }
                outfile << "}" << endl;
            }
            outfile << "clustering_time: "<< (double)tClustering << endl;
        }

        //Silhouette
        for(int i=0; i<snumbers.size(); i++){
            outfile << "Silhouette: [";
            if(i!=snumbers.size()-1){
                outfile << snumbers[i] << ", ";
            }
            else outfile << snumbers[i] << "]\n";
        }

        //Objective Function
        outfile << "Value of Objective Function: "<< objectiveSum << "\n\n";

        outfile.close();
    }
    else cout << "Unable to open outout file" << endl;
}

Clusters::~Clusters(){}

void updateDataset(Dataset *trainSet, char *I, int numBytes){
    fstream trainInput(I);
    if(!trainInput.is_open()){
        cerr<<"Failed to open input data."<<endl;
        return;
    }
    unsigned short *tmp = trainSet->imageAt(0);
    unsigned short bigEndShort = 0, littleEndShort = 0;
    for (int i=0; i<trainSet->getNumberOfPixels()*trainSet->getNumberOfImages(); i++){
        bigEndShort = 0;
        trainInput.read((char*)&bigEndShort, numBytes);
        littleEndShort = (bigEndShort>>8) | ((bigEndShort<<8));
        if (numBytes==1) memcpy(tmp, &bigEndShort, sizeof(littleEndShort));
        else memcpy(tmp, &littleEndShort, sizeof(littleEndShort));
        tmp++;
    }
    trainInput.close();
}

vector<vector<int>> readClassFile(char *n){
    vector<vector<int>> images;
    fstream classesFile(n);
    if(!classesFile.is_open()){
        cerr<<"Failed to open " << n <<endl;
        return images;
    }
    string line;
    int points=0;
    while(getline(classesFile, line, '\n')){
        images.push_back(std::vector<int>());
        string delimiter = "CLUSTER-";
        string token = line.substr(line.find(delimiter)+delimiter.length(), line.find(delimiter)+delimiter.length());
        int clusternum = (int)line.at(line.find(delimiter)+delimiter.length())-48;
        delimiter = "size: ";
        line.pop_back();
        token = line.substr(line.find(delimiter)+delimiter.length(), line.size()-1);
        istringstream is_line(token);
        string imagenum;
        int indexat=0, clustersize=0;
        while(getline(is_line, imagenum, ',')){
            if(indexat!=0) images[clusternum].push_back(stoi(imagenum));
            else{
                clustersize=stoi(imagenum);
                points+=clustersize;
            }
            indexat++;
        }
    }
    classesFile.close();
    return images;
}
