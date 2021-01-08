#ifndef DATASET_HPP
#define DATASET_HPP
class Dataset{
    int magicNumber;
    int numberOfImages;
    int numberOfRows;
    int numberOfColumns;
    int numberOfPixels;

    unsigned char* images;
    public:
        Dataset(int, int, int, int);
        ~Dataset();
        int getNumberOfRows(){return numberOfRows;};
        int getNumberOfColumns(){return numberOfColumns;};
        int getMagicNumber(){return magicNumber;};
        int getNumberOfImages(){return numberOfImages;};
        int getNumberOfPixels(){return numberOfPixels;};
        inline unsigned char* imageAt(int index){
            return &images[index*numberOfPixels];
        }
        inline int dimension(){
            return numberOfRows*numberOfColumns;
        }
};
#endif
