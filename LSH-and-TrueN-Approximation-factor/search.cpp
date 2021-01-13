#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "hash.hpp"
#include "dataset.hpp"
#include "lshAlgorithms.hpp"
#include "metrics.hpp"
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
// ./search -d ../Datasets/train-images-idx3-ubyte -i ../Datasets/reduced_dims_trainset -q ../Datasets/t10k-images-idx3-ubyte -s ../Datasets/reduced_dims_testset -k 3 -L 4 -o fileoutlsh

using namespace std;

int main(int argc, char** argv){
    if (argc>6 && argc<16){
        char *d=NULL, *q=NULL, *o=NULL, *k=NULL, *l=NULL, *n=NULL, *r=NULL, *in=NULL, *s=NULL;
        double R=1.0, exec_time;
        int K=4, L=5, N=1;
        bool vars[7] = { 0 };
        for (int i=0; i<7; i++) vars[i] = true;
        for (int i = 0; i<argc; i++){
            if (!strcmp("-d", argv[i]) && vars[0]) {
                d = (char*)argv[i+1];   /* -inputfile */
                vars[0] = false;
            }
            if (!strcmp("-i", argv[i]) && vars[1]) {
                in = (char*)argv[i+1];   /* -inputfile */
                vars[1] = false;
            }
            if (!strcmp("-q", argv[i]) && vars[2]) {
                q = (char*)argv[i+1];   /* -queryfile */
                vars[2] = false;
            }
            if (!strcmp("-s", argv[i]) && vars[3]) {
                s = (char*)argv[i+1];   /* -inputfile */
                vars[3] = false;
            }
            if (!strcmp("-k", argv[i]) && vars[4]) {
                k = (char*)argv[i+1];   /* -k */
                vars[4] = false;
            }
            if (!strcmp("-L", argv[i]) && vars[5]) {
                l = (char*)argv[i+1];   /* -L */
                vars[5] = false;
            }
            if (!strcmp("-o", argv[i]) && vars[6]) {
                o = (char*)argv[i+1];   /* -o */
                vars[6] = false;
            }
        }

        if(d==NULL || q==NULL || o==NULL || in==NULL || s==NULL){
            cout << "You must run the program with parameters(REQUIRED): –d <input file> –q <query file>" << endl;
            cout << "With additional parameters: –k <int> -L <int> -ο <output file>" << endl;
            return(0);
        }
        else{
            if (k!=NULL) K = atoi(k);
            if (l!=NULL) L = atoi(l);
            unsigned int magicNumber = 0,numberOfImages = 0,numberOfRows = 0,numberOfColumns = 0, img=0;
            int numOfpixels;

            //Open train file
            ifstream trainInput;

            trainInput.open(d, std::ios::binary);
            if(!trainInput.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return(0);
            }
            trainInput.read((char*)&magicNumber, 4);
            trainInput.read((char*)&numberOfImages, 4);
            trainInput.read((char*)&numberOfRows, 4);
            trainInput.read((char*)&numberOfColumns, 4);

            //Convert intergers from Big Endian to Little Endian
            magicNumber = SWAP_INT32(magicNumber);
            numberOfImages = SWAP_INT32(numberOfImages);
            numberOfRows = SWAP_INT32(numberOfRows);
            numberOfColumns = SWAP_INT32(numberOfColumns);

            Dataset trainSet(magicNumber, numberOfImages, numberOfColumns, numberOfRows);
            unsigned short* temp = trainSet.imageAt(0);
            unsigned short bigEndShort = 0, littleEndShort = 0;
            for(int i = 0; i < ((trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages())); i++){
                bigEndShort = 0;
                trainInput.read((char*)&bigEndShort, 1);
                littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                memcpy(temp, &bigEndShort, sizeof(littleEndShort));
                temp++;
            }    
            //trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
            trainInput.close();
            // cout << "trainSet: Images:" << numberOfImages << " Rows: "<< numberOfRows << " Columns: "<< numberOfColumns << " 152nd Pixel at image 0 "<< trainSet.imageAt(0)[152]<< endl;

            ifstream trainInputR;
            
            trainInputR.open(in, std::ios::binary);
            if(!trainInputR.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return(0);
            }
            trainInputR.read((char*)&magicNumber, 4);
            trainInputR.read((char*)&numberOfImages, 4);
            trainInputR.read((char*)&numberOfRows, 4);
            trainInputR.read((char*)&numberOfColumns, 4);
            // cout << numberOfImages << endl;
            //Convert intergers from Big Endian to Little Endian
            magicNumber = SWAP_INT32(magicNumber);
            numberOfImages = SWAP_INT32(numberOfImages);
            numberOfRows = SWAP_INT32(numberOfRows);
            numberOfColumns = SWAP_INT32(numberOfColumns);
            // cout << numberOfImages << endl;

            Dataset trainSetR(magicNumber, numberOfImages, numberOfColumns, numberOfRows);
            temp = trainSetR.imageAt(0);
            bigEndShort = 0; 
            littleEndShort = 0;
            for(int i = 0; i < ((trainSetR.getNumberOfPixels())*(trainSetR.getNumberOfImages())); i++){
                trainInputR.read((char*)&bigEndShort, 2);
                littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                memcpy(temp, &littleEndShort, sizeof(littleEndShort));
                temp++;
            }    
            //trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
            trainInputR.close();
            // cout << "trainSetR: Images:" << numberOfImages << " Rows: "<< numberOfRows << " Columns: "<< numberOfColumns << " 2nd Pixel at image 1 "<< trainSetR.imageAt(1)[1]<< endl;

            // cout << trainSet.imageAt(0)[1] <<endl;
            img = numberOfImages;
            string queryfile, outputfile, answer,reducedqueryfile;
            bool termination = false;

            while(!termination){
                /* PROGRAM STARTS HERE */
                clock_t tStart = clock();

                //Open query file
                fstream queryInput(q);
                if(!queryInput.is_open()){
                    cerr<<"Failed to open input data."<<endl;
                    return 0;
                }
                queryInput.read((char*)&magicNumber, 4);
                queryInput.read((char*)&numberOfImages, 4);
                queryInput.read((char*)&numberOfRows, 4);
                queryInput.read((char*)&numberOfColumns, 4);

                //Convert intergers from Big Endian to Little Endian
                magicNumber = SWAP_INT32(magicNumber);
                numberOfImages = SWAP_INT32(numberOfImages);
                numberOfRows = SWAP_INT32(numberOfRows);
                numberOfColumns = SWAP_INT32(numberOfColumns);
                Dataset querySet(magicNumber, numberOfImages, numberOfColumns, numberOfRows);

                temp = querySet.imageAt(0);
                bigEndShort = 0;
                littleEndShort = 0;
                for(int i = 0; i < ((querySet.getNumberOfPixels())*(querySet.getNumberOfImages())); i++){
                    bigEndShort = 0;
                    queryInput.read((char*)&bigEndShort, 1);
                    littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                    memcpy(temp, &bigEndShort, sizeof(littleEndShort));
                    temp++;
                }    
                queryInput.close();
                // cout << "querySet: Images:" << numberOfImages << " Rows: "<< numberOfRows << " Columns: "<< numberOfColumns << " 2nd Pixel at image 1 "<< querySet.imageAt(1)[1]<< endl;
                fstream queryInputR(s);
                if(!queryInputR.is_open()){
                    cerr<<"Failed to open input data."<<endl;
                    return 0;
                }
                queryInputR.read((char*)&magicNumber, 4);
                queryInputR.read((char*)&numberOfImages, 4);
                queryInputR.read((char*)&numberOfRows, 4);
                queryInputR.read((char*)&numberOfColumns, 4);

                //Convert intergers from Big Endian to Little Endian
                magicNumber = SWAP_INT32(magicNumber);
                numberOfImages = SWAP_INT32(numberOfImages);
                numberOfRows = SWAP_INT32(numberOfRows);
                numberOfColumns = SWAP_INT32(numberOfColumns);
                Dataset querySetR(magicNumber, numberOfImages, numberOfColumns, numberOfRows);

                temp = querySetR.imageAt(0);
                bigEndShort = 0;
                littleEndShort = 0;
                for(int i = 0; i < ((querySetR.getNumberOfPixels())*(querySetR.getNumberOfImages())); i++){
                    queryInputR.read((char*)&bigEndShort, 2);
                    littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                    memcpy(temp, &littleEndShort, sizeof(littleEndShort));
                    temp++;
                }    
                queryInputR.close();
                // cout << "querySetR: Images:" << numberOfImages << " Rows: "<< numberOfRows << " Columns: "<< numberOfColumns << " 2nd Pixel at image 1 "<< querySetR.imageAt(1)[1]<< endl;

                ///////////////////////////////////////structure test///////////////////////////////////////
                int bucketsNumber = floor(trainSet.getNumberOfImages()/16);

                // int W = FindW(img, &trainSet);
                // cout << "W is " << W << endl;
                int W = 40000;
                HashFunction** hashFamily = new HashFunction*[trainSet.getNumberOfPixels()];
                for(int i = 0; i < trainSet.getNumberOfPixels();i++){
                    hashFamily[i] = NULL;
                }

                HashTable **hashTables = new HashTable*[L];
                for(int i=0; i<L; i++){
                    hashTables[i] = new HashTable(trainSet.getNumberOfPixels(),bucketsNumber, K,W,hashFamily);

                    for(int j=0; j<img; j++){
                        unsigned int g_hash = (unsigned int)(hashTables[i]->ghash(trainSet.imageAt(j)));
                        hashTables[i]->getBucketArray()[g_hash%bucketsNumber]->addImage(j,g_hash,trainSet.imageAt(j));
                    }
                }

                ofstream outputf(o);
                if (!outputf.is_open()){
                    cerr<<"Failed to open output data."<<endl;
                    return 0;
                }
                clock_t lshAnnStart, lshRngStart, AnnTrueStart, RngTrueStart,AnnTrueRStart;
                double lshAnnTime, lshRngTime, trueAnnTime, trueRngTime, trueAnnRTime;
                double approx_lsh = 0, approx_reduced = 0, tempReduced = 0, tempLsh = 0;
                for(int index=0; index<numberOfImages; index++){
                    // cout << index << endl;
                    vector<Neighbor> ANNneighbors;
                    vector<Neighbor> TrueNeighbors;
                    vector<Neighbor> True_Reduced_Neighbors;
                    // vector<Neighbor> RNGneighbors;

                    lshAnnStart = clock();
                    ANNsearch(ANNneighbors,L, N, querySet.imageAt(index), hashTables);
                    lshAnnTime = lshAnnTime + (double)(clock() - lshAnnStart)/CLOCKS_PER_SEC;

                    /////////changes for Reduced//////////////

                    AnnTrueRStart = clock();
                    trueDistanceWithNeighbors(True_Reduced_Neighbors, querySetR.imageAt(index), &trainSetR,trainSetR.getNumberOfPixels());
                    trueAnnRTime = trueAnnRTime + (double)(clock() - AnnTrueStart)/CLOCKS_PER_SEC;

                    ///////////////////////////////////////////

                    AnnTrueStart = clock();
                    trueDistanceWithNeighbors(TrueNeighbors, querySet.imageAt(index), &trainSet, trainSet.getNumberOfPixels());
                    trueAnnTime = trueAnnTime + (double)(clock() - AnnTrueStart)/CLOCKS_PER_SEC;


                    int ANNsize = ANNneighbors.size();
                    int TrueSize = TrueNeighbors.size();
                    int TrueRsize = True_Reduced_Neighbors.size();
                    outputf << "\nQuery: " << index << endl;
                
                    if((TrueRsize) > 0){
                        outputf << "Nearest neighbor Reduced: " << True_Reduced_Neighbors[True_Reduced_Neighbors.size()-1].getIndex() << endl;
                    }
                    if((ANNsize) > 0){
                        outputf << "Nearest neighbor LSH: " << ANNneighbors[ANNneighbors.size()-1].getIndex() << endl;   
                    } 
                    if((TrueSize) > 0){
                        outputf << "Nearest neighbor True: " << TrueNeighbors[TrueNeighbors.size()-1].getIndex() << endl;
                    }
                    if((TrueRsize) > 0){
                        tempReduced =  manhattan(querySet.imageAt(index), trainSet.imageAt(True_Reduced_Neighbors[True_Reduced_Neighbors.size()-1].getIndex()), trainSet.getNumberOfPixels());
                        outputf << "distanceReduced: " << tempReduced << endl;
                    }
                    if((ANNsize) > 0){
                        tempLsh = ANNneighbors[ANNneighbors.size()-1].getDist();
                        outputf << "distanceLSH: " << tempLsh << endl;   
                    } 
                    if((TrueSize) > 0){
                        approx_reduced = approx_reduced + (tempReduced/TrueNeighbors[TrueNeighbors.size()-1].getDist());
                        approx_lsh =  approx_lsh + (tempLsh/TrueNeighbors[TrueNeighbors.size()-1].getDist());

                        outputf << "distanceTrue: " << TrueNeighbors[TrueNeighbors.size()-1].getDist() << endl;
                    }  
                }
                outputf << "\ntReduced: " << trueAnnRTime/numberOfImages << endl;
                outputf << "tLSH: " << lshAnnTime/numberOfImages << endl;
                outputf << "tTrue: " << trueAnnTime/numberOfImages << endl;
                outputf << "Approximation Factor LSH: " <<  (double)approx_lsh/(double)numberOfImages << endl;
                outputf << "Approximation Factor Reduced: " << (double)approx_reduced/(double)numberOfImages<< endl;
                
                outputf.close();

                for(int i=0; i<trainSet.getNumberOfPixels(); i++){
                    if(hashFamily[i]!=NULL){
                        delete hashFamily[i];
                    }
                }
                delete[] hashFamily;
                for(int i=0; i<L; i++) delete hashTables[i];
                delete[] hashTables;

                /* PROGRAM ENDS HERE */
                exec_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
                cout << "\nExecution time is: "<< exec_time << endl;

                cout << "\nYou want to execute the lsh with another queryfile? (Y/N)" << endl;
                cin >> answer;
                if(answer.compare("Y")==0 || answer.compare("Yes")==0) {
                    cout << "Please type the path for queryfile:" << endl;
                    cin >> queryfile;
                    q = &queryfile[0];

                    cout << "Please type the path for new dimensions queryfile:" << endl;
                    cin >> reducedqueryfile;
                    s = &reducedqueryfile[0];

                    cout << "Please type the path for outputfile:" << endl;
                    cin >> outputfile;
                    o = &outputfile[0];
                }
                else if (answer.compare("N")==0 || answer.compare("No")==0){
                    termination = true;
                    cout << "The program will terminate." << endl;
                }
                else {
                    cout << "This answer is not recognizable. The program will terminate." << endl;
                    termination = true;
                }


            }
        }

    }
    else {
        cout << "You must run the program with parameters(REQUIRED): –d <input file> –q <query file>" << endl;
        cout << "With additional parameters: –k <int> -L <int> -ο <output file> " << endl;
    }
}
