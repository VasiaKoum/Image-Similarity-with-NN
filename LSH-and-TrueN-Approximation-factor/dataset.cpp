#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <cstring>
#include <time.h>
#include "dataset.hpp"

using namespace std;

Dataset::Dataset(int magic,int imgs, int columns,int rows) : magicNumber(magic), numberOfColumns(columns), numberOfRows(rows), numberOfImages(imgs){
    this->numberOfPixels = columns*rows;
    this->images = new unsigned short[imgs*numberOfPixels];
}

Dataset::~Dataset(){
    delete[] images;
}
