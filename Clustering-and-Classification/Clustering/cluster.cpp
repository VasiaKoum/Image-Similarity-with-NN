#include <string.h>
#include <sstream>
#include "centroids.hpp"
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
using namespace std;

int main(int argc, char** argv){
    if (argc>6 && argc<16){
        char *I=NULL, *c=NULL, *o=NULL, *n=NULL, *d=NULL;
        double R=1.0, exec_time;
        int K=-1, L=3, kLSH=14, M=10, kHYP=3, probes=2;
        bool vars[5] = { 0 };
        for (int i=0; i<5; i++) vars[i] = true;
        for (int i=0; i<argc; i++){
            if (!strcmp("-i", argv[i]) && vars[0]) {
                I = (char*)argv[i+1];   /* -input file new space*/
                vars[0] = false;
            }
            if (!strcmp("-c", argv[i]) && vars[1]) {
                c = (char*)argv[i+1];   /* -configuration file */
                vars[1] = false;
            }
            if (!strcmp("-o", argv[i]) && vars[2]) {
                o = (char*)argv[i+1];   /* -output file */
                vars[2] = false;
            }
            if (!strcmp("-d", argv[i]) && vars[3]) {
                d = (char*)argv[i+1];   /* -input file original space */
                vars[3] = false;
            }
            if (!strcmp("-n", argv[i]) && vars[4]) {
                n = (char*)argv[i+1];   /* classes from NN as clusters */
                vars[4] = false;
            }
        }

        if(I==NULL || c==NULL || o==NULL || d==NULL || n==NULL){
            cout << "You must run the program with parameters: -d <input file original space> -i <input file new space> -n <classes from NN as clusters> -c <configuration file> -o <output file>" << endl;
            return 0;
        }
        else{

            /* PROGRAM STARTS HERE */
            clock_t tStart = clock();

            unsigned int magicNumber = 0, numberOfImages = 0, numberOfRows = 0, numberOfColumns = 0, img=0;

            //Open train file
            fstream trainInput(I);
            if(!trainInput.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return 0;
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
            trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
            trainInput.close();

            //Open configuration file
            fstream configuration(c);
            if(!configuration.is_open()){
                cerr<<"Failed to open configuration file."<<endl;
                return 0;
            }
            else{
                string line;
                int i=0;
                while(getline(configuration, line, '\n')){
                    istringstream is_line(line);
                    string type;
                    if(getline(is_line, type, ' ')){
                        string number;
                        if(getline(is_line, number, ' ')){
                            if(number[number.size()-1] == '\r' || number[number.size()-1] == '\n') {
                                number.erase(number.size() - 1);
                            }
                            if(type.compare("number_of_clusters:")==0) K = stoi(number);
                            else if(type.compare("number_of_vector_hash_tables:")==0) L = stoi(number);
                            else if(type.compare("number_of_vector_hash_functions:")==0) kLSH = stoi(number);
                            else if(type.compare("max_number_M_hypercube:")==0) M = stoi(number);
                            else if(type.compare("number_of_hypercube_dimensions:")==0) kHYP = stoi(number);
                            else if(type.compare("number_of_probes:")==0) probes = stoi(number);
                        }
                    }
                }
            }
            configuration.close();

            if(K<0){
                cout << "Configuration file must contain number_of_clusters" << endl;
                cout << "Program terminates." << endl;
                return 0;
            }

            Centroids centroids(K, numberOfImages, &trainSet);
            centroids.Initialize();
            Clusters clusters(&centroids);
            clusters.Clustering(o);

            /* PROGRAM ENDS HERE */
            exec_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
            cout << "\nExecution time is: "<< exec_time << endl;
        }
    }
    else{
cout << "You must run the program with parameters: -d <input file original space> -i <input file new space> -n <classes from NN as clusters> -c <configuration file> -o <output file>" << endl;
    }
}
