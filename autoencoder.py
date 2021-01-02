# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte -q ./Datasets/t10k-images-idx3-ubyte -od outdata -oq outquery
import sys
import time
import struct
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from functions import *

# Autoencoder hyperparameters -> number of layers, filter size, number of filters/layer, number of epochs, batch size
def main():
    if (len(sys.argv) != 9):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    if ("-d" in sys.argv and "-q" in sys.argv and "-od" in sys.argv and "-oq" in sys.argv):
        dataset = sys.argv[sys.argv.index("-d")+1]
        queryset = sys.argv[sys.argv.index("-q")+1]
        output_data = sys.argv[sys.argv.index("-od")+1]
        output_query = sys.argv[sys.argv.index("-oq")+1]
    else:
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset -q queryset -od output_data -oq output_query")

    # numarray[0] -> magic_number, [1] -> images, [2] -> rows, [3] -> columns
    df = values_df()
    hypernames = ["Layers", "Filter_Size", "Filters/Layer", "Epochs", "Batch_Size"]
    pixels, numarray = numpy_from_dataset(dataset, 4)
    if (len(numarray)!=4 or len(pixels)==0):
        sys.exit("Input dataset does not have the required number of values")
    train_X, valid_X, train_Y, valid_Y = reshape_dataset(pixels, numarray)
    print("Data ready in numpy array!\n")
    # Layers, Filter_size, Filters/Layer, Epochs, Batch_size
    parameters = [2, 3, 4, 1, 128]
    # parameters = input_parameters()
    newparameter = [[] for i in range(len(parameters))]
    originparms = parameters.copy()
    oldparm = -1

    while (True):
        print("\nBegin building model...")
        input_img = Input(shape=(numarray[2], numarray[3], 1))
        encoder_layer = encoder(input_img, parameters)
        bottleneck_layer = bottleneck(encoder_layer, parameters)

        autoencoder = Model(input_img, decoder(bottleneck_layer, parameters))
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

        train_time = time.time()
        autoencoder_train = autoencoder.fit(train_X, train_Y, batch_size=parameters[4], epochs=parameters[3], verbose=1, validation_data=(valid_X, valid_Y))
        train_time = time.time() - train_time
        # print(autoencoder.summary())

        embedding_outputs = (K.function([autoencoder.input], [layer.output])(train_X) for layer in autoencoder.layers if layer.output_shape == (None, 10))
        lst = list(embedding_outputs)
        # newlst = [(i/255).astype(int) for i in lst[0][0]]
        newlst = [(i).astype(int) for i in lst[0][0]]

        write_output(newlst, output_data)
        read_lst = read_output(output_data)

        # User choices:
        parameters, continue_flag, oldparm = user_choices(autoencoder, autoencoder_train, parameters, originparms, train_time, newparameter, oldparm, df, hypernames)
        if (not continue_flag):
            break;

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
