#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "dataset.hpp"
#include "Algorithms.hpp"
#include "metrics.hpp"
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
// ./lsh -d ../Datasets/train-images-idx3-ubyte -i ../Datasets/reduced_dims_trainset -q ../Datasets/t10k-images-idx3-ubyte -s ../Datasets/reduced_dims_testset -k 3 -L 4 -o fileoutlsh
using namespace std;

int main(int argc, char** argv){
    if (argc>6 && argc<12){
        char *d=NULL, *q=NULL, *o=NULL, *l1=NULL, *l2=NULL;
        double exec_time;
        bool vars[6] = { 0 };
        for (int i=0; i<6; i++) vars[i] = true;
        for (int i = 0; i<argc; i++){
            if (!strcmp("-d", argv[i]) && vars[0]) {
                d = (char*)argv[i+1];   /* -inputfile */
                vars[0] = false;
            }
            if (!strcmp("-l1", argv[i]) && vars[1]) {
                l1 = (char*)argv[i+1];   /* -inputfile */
                vars[1] = false;
            }
            if (!strcmp("-q", argv[i]) && vars[2]) {
                q = (char*)argv[i+1];   /* -queryfile */
                vars[2] = false;
            }
            if (!strcmp("-l2", argv[i]) && vars[3]) {
                l2 = (char*)argv[i+1];   /* -inputfile */
                vars[3] = false;
            }
            if (!strcmp("-o", argv[i]) && vars[4]) {
                o = (char*)argv[i+1];   /* -o */
                vars[4] = false;
            }
            if (!strcmp("-EMD", argv[i]) && vars[5]) {
                vars[5] = false;
            }
        }

        if(d==NULL || q==NULL || o==NULL || l1==NULL || l2==NULL){
            cout << "You must run the program with correct parameters" << endl;
            return(0);
        }
        else{
            unsigned int magicNumber = 0,numberOfImages = 0,numberOfRows = 0,numberOfColumns = 0, img=0;
            int numOfpixels;

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

            numberOfImages = 100;

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

            ifstream trainLabelsInput;
            trainLabelsInput.open(l1, std::ios::binary);
            if(!trainLabelsInput.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return(0);
            }

            trainLabelsInput.read((char*)&magicNumber, 4);
            trainLabelsInput.read((char*)&numberOfImages, 4);
            //Convert intergers from Big Endian to Little Endian
            magicNumber = SWAP_INT32(magicNumber);
            numberOfImages = SWAP_INT32(numberOfImages);

            Dataset trainLabels(magicNumber, numberOfImages, 1, 1);
            temp = trainLabels.imageAt(0);
            bigEndShort = 0, littleEndShort = 0;
            for(int i = 0; i < (trainLabels.getNumberOfImages()); i++){
                bigEndShort = 0;
                trainLabelsInput.read((char*)&bigEndShort, 1);
                littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                memcpy(temp, &bigEndShort, sizeof(littleEndShort));
                temp++;
            }    
            //trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
            trainLabelsInput.close();
            
            img = numberOfImages;
            string queryfile, outputfile, answer;
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

                ifstream queryLabelsInput;
                queryLabelsInput.open(l2, std::ios::binary);
                if(!queryLabelsInput.is_open()){
                    cerr<<"Failed to open input data."<<endl;
                    return(0);
                }

                queryLabelsInput.read((char*)&magicNumber, 4);
                queryLabelsInput.read((char*)&numberOfImages, 4);
                //Convert intergers from Big Endian to Little Endian
                magicNumber = SWAP_INT32(magicNumber);
                numberOfImages = SWAP_INT32(numberOfImages);

                Dataset queryLabels(magicNumber, numberOfImages, 1,1);
                temp = queryLabels.imageAt(0);
                bigEndShort = 0, littleEndShort = 0;
                for(int i = 0; i < (queryLabels.getNumberOfImages()); i++){
                    bigEndShort = 0;
                    queryLabelsInput.read((char*)&bigEndShort, 1);
                    littleEndShort = (bigEndShort>>8) | (bigEndShort<<8);
                    memcpy(temp, &bigEndShort, sizeof(littleEndShort));
                    temp++;
                }    
                //trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
                queryLabelsInput.close();

                ///EMD
                string str = "python3 linear.py";
                str = str + " -d " +  d;
                str = str + " -q " +  q;
                str = str + " -l1 " + l1;
                str = str + " -l2 " + l2;
                str = str + " -o " +  o;   
                const char *command = str.c_str(); 
                cout << command << endl; 
                system(command); 

                ofstream outputf(o);
                if (!outputf.is_open()){
                    cerr<<"Failed to open output data."<<endl;
                    return 0;
                }
                //MANHATTAN
                clock_t AnnTrueStart;
                double trueAnnTime;
                int success = 0;
                double average_acc = 0;

                numberOfImages = 10;

                for(int index=0; index<numberOfImages; index++){
                    vector<Neighbor> TrueNeighbors;

                    AnnTrueStart = clock();
                    trueDistanceWithNeighbors(TrueNeighbors, querySet.imageAt(index), &trainSet, trainSet.getNumberOfPixels());
                    sort(TrueNeighbors.begin(), TrueNeighbors.end(), compareNeighbor); 
                    trueAnnTime = trueAnnTime + (double)(clock() - AnnTrueStart)/CLOCKS_PER_SEC;

                    int TrueSize = TrueNeighbors.size();
  
                    if((TrueSize) > 0){
                        // outputf << "Nearest neighbor True: " << TrueNeighbors[TrueNeighbors.size()-1].getIndex() << endl;
                    }
                    if((TrueSize) > 0){
                        // outputf << "distanceTrue: " << TrueNeighbors[TrueNeighbors.size()-1].getDist() << endl;
                        success = 0;
                        for(int i=0; i<10; i++){
                            if((unsigned short)*(trainLabels.imageAt(TrueNeighbors[TrueNeighbors.size()-1-i].getIndex())) == (unsigned short)*(queryLabels.imageAt(index))){
                                success++;
                            }
                        }
                    }
                    average_acc += (double)success/10.0;       
                }
                outputf << "Average Correct Search Results MANHATTAN: " << average_acc/(double)numberOfImages << endl;
                cout << "Average Correct Search Results MANHATTAN: " << average_acc/(double)numberOfImages << endl;
                outputf.close();
                exec_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
                cout << "MANHATTAN execution time is: "<< exec_time << endl;

                cout << "\nYou want to execute the lsh with another queryfile? (Y/N)" << endl;
                cin >> answer;
                if(answer.compare("Y")==0 || answer.compare("Yes")==0) {
                    cout << "Please type the path for queryfile:" << endl;
                    cin >> queryfile;
                    q = &queryfile[0];
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
        cout << "You must run the program with correct parameters" << endl;
    }
}