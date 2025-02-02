import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from sklearn.metrics import classification_report
from tensorflow.python.keras import backend as K
from PIL import Image

def numpy_from_dataset(inputpath, numbers, per_2_bytes):
    pixels = []
    numarray = []
    with open(inputpath, "rb") as file:
        for x in range(numbers):
            numarray.append(int.from_bytes(file.read(4), byteorder='big'))
        print("Storing data in array...")
        # 2d numpy array for images->pixels
        if numbers == 4:
            if per_2_bytes:
                data = file.read(2)
                while data:
                    pixels.append(int.from_bytes(data, byteorder='big'))
                    data = file.read(2)
                pixels = np.array(list(bytes_group(numarray[3], pixels, fillvalue=0)))
            else:
                pixels = np.array(list(bytes_group(numarray[2]*numarray[3], file.read(), fillvalue=0)))
        elif numbers == 2:
            pixels = np.array(list(bytes_group(1, file.read(), fillvalue=0)))
    return pixels, numarray

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

def encoder(input_img, parameters):
    layers = parameters[0]
    filter_size = parameters[1]
    filters = parameters[2]
    conv = input_img
    for i in range(layers):
        conv = Conv2D(filters, (filter_size, filter_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if (i<2):
            conv = MaxPooling2D(pool_size=(2,2))(conv)
        # conv = Dropout(0.2)(conv)
        filters*=2
    return conv

def bottleneck(enc, parameters):
    flatten_layer = Flatten()(enc)
    embedding_layer = Dense(parameters[5])(flatten_layer)
    dense_layer = Dense(flatten_layer.shape[1])(embedding_layer)
    reshape_layer = Reshape((enc.shape[1], enc.shape[2], enc.shape[3]))(dense_layer)
    return reshape_layer, embedding_layer

def decoder(conv, parameters):
    layers = parameters[0]
    filter_size = parameters[1]
    filters = parameters[2]*pow(2,parameters[0]-1)
    for i in range(layers):
        conv = Conv2DTranspose(filters, (filter_size, filter_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if (i>=layers-2):
            conv = UpSampling2D((2,2))(conv)
        filters/=2
    conv = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(conv)
    return conv

def save_model(model):
    modelname = input("Type the name for model(without extension eg.h5): ")
    print("Saving Model: "+modelname+".json & "+modelname+".h5...")
    # Save model in JSON file
    model_json = model.to_json()
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
    # Save weights from model in h5 file
    model.save_weights(modelname+".h5")

def load_model(modelname):
    # Load model from JSON file
    print("Loading Model: "+modelname+".json & "+modelname+".h5...")
    json_file = open(modelname+".json", 'r')
    autoencoder_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(autoencoder_json)
    # Load weights from h5 file
    autoencoder.load_weights(modelname+".h5")
    return autoencoder

def error_graphs(modeltrain, parameters, train_time, newparameter, indexparm, originparms, hypernames):
    loss = []
    val = []
    values = []
    times = []
    for i in range(len(newparameter)):
        loss.clear()
        val.clear()
        times.clear()
        values.clear()
        for j in newparameter[i]:
            values.append(j[0])
            loss.append(j[1])
            val.append(j[2])
            times.append(j[3])
        if (i == indexparm-1):
            values.append(parameters[indexparm-1])
            loss.append(modeltrain.history['loss'][-1])
            val.append(modeltrain.history['val_loss'][-1])
            times.append(train_time)
        if newparameter[i]:
            graphname = name_parameter(originparms, i, True, hypernames) + ".png"
            plt.plot(values, loss, label='train', linestyle='dashed', linewidth = 3,  marker='o', markersize=9)
            plt.plot(values, val, label='test', linestyle='dashed', linewidth = 3,  marker='o', markersize=9)
            plt.title('Loss / Mean Squared Error in '+str(round(times[-1], 3))+'sec')
            plt.ylabel('Loss')
            plt.xlabel(name_parameter(originparms, i, False, hypernames))
            plt.legend(['loss', 'val_loss'], loc='upper left')
            print("Save graph with name: ",graphname)
            plt.savefig(graphname)
            plt.show()
            plt.close()
    return

def name_parameter(parameters, number, flag, hypernames):
    name = ""
    if (flag):
        if (number==0):
            name = "Lx"+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])+"_LV"+str(parameters[5])
        elif (number==1):
            name = "L"+str(parameters[0])+"_FSx"+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])+"_LV"+str(parameters[5])
        elif (number==2):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FLx"+"_E"+str(parameters[3])+"_B"+str(parameters[4])+"_LV"+str(parameters[5])
        elif (number==3):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_Ex"+"_B"+str(parameters[4])+"_LV"+str(parameters[5])
        elif (number==4):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_Bx"+"_LV"+str(parameters[5])
        elif (number==5):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])+"_LVx"
    else:
        name = hypernames[number]
    return name

def reshape_dataset(dataset, numarray):
    train_X, valid_X, train_Y, valid_Y = train_test_split(dataset, dataset, test_size=0.2, random_state=13)
    # Reshapes to (x, rows, columns)
    train_X = np.reshape(train_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_X = np.reshape(valid_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    train_Y = np.reshape(train_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_Y = np.reshape(valid_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    return train_X, valid_X, train_Y, valid_Y

def user_choices(model, modeltrain, parameters, originparms, train_time, newparameter, oldparm, df, hypernames):
    continue_flag = True
    while (True):
        try:
            run_again = int(input("\nUSER CHOICES: choose one from below options(1-4): \n1)Execute program with different hyperparameter\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n---------------> "))
        except:
            print("Invalid choice.Try again\n")
            continue
        if (run_again==1):
            try:
                indexparm = int(input("Choose what parameter would like to change (options 1-6): \n1)Layers\n2)Filter size\n3)Filters/Layer\n4)Epochs\n5)Batch size\n6)Latent vector\n---------------> "))
            except:
                print("Invalid choice.Try again\n")
                continue
            if (indexparm>=1 and indexparm<=6):
                try:
                    changepar = int(input("Number for "+ name_parameter(parameters, indexparm-1, False, hypernames) +" is "+str(parameters[indexparm-1])+". Type the new number: "))
                except:
                    print("Invalid choice.Try again\n")
                    continue
                tmpparm = oldparm
                if tmpparm<0:
                    tmpparm = indexparm
                tmp = [parameters[tmpparm-1]] + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]] + [train_time]
                newparameter[tmpparm-1].append(tmp)
                df.loc[len(df), :] = parameters + [train_time] + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]]
                parameters = originparms.copy()
                parameters[indexparm-1] = changepar
                oldparm = indexparm
                break
            else:
                print("Invalid choice.Try again\n")
        elif (run_again == 2):
            error_graphs(modeltrain, parameters, train_time, newparameter, oldparm, originparms, hypernames)
            # continue_flag = False
            # break
        elif (run_again == 3):
            save_model(model)
            # continue_flag = False
            # break
        elif (run_again == 4):
            df.loc[len(df), :] = parameters + [train_time] + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]]
            df.drop_duplicates(subset=['Layers', 'Filter_Size', 'Filters/Layer', 'Epochs', 'Batch_Size', 'Latent_vector'], inplace=True)
            df = df.sort_values(by = 'Val_Loss', ascending=True)
            df.to_csv('loss_values.csv', sep='\t', index=False)
            continue_flag = False
            print("Program terminates...\n")
            break
        else:
            print("Invalid choice.Try again\n")
    return parameters, continue_flag, oldparm;

def input_parameters():
    parameters = []
    try:
        parameters.append(int(input("Type number of layers: ")))
        parameters.append(int(input("Type filter size: ")))
        parameters.append(int(input("Type number of filters/layer: ")))
        parameters.append(int(input("Type number of epochs: ")))
        parameters.append(int(input("Type batch size: ")))
        parameters.append(int(input("Type latent vector size: ")))
    except:
        print("Invalid choice.Try again\n")
    return parameters

def values_df():
    try:
        df = pd.read_csv('loss_values.csv',sep='\t')
    except:
        loss_values = {'Layers': [], 'Filter_Size': [], 'Filters/Layer': [], 'Epochs': [], 'Batch_Size': [], 'Latent_vector': [], 'Train_Time': [], 'Loss': [], 'Val_Loss': []}
        df = pd.DataFrame(data=loss_values)
    return df

def write_output(list_output, numdata, filename):
    output_file = open(filename, 'wb')
    rows = 1
    columns = 10
    output_file.write(rows.to_bytes(4, 'big'))
    output_file.write(numdata.to_bytes(4, 'big'))
    output_file.write(rows.to_bytes(4, 'big'))
    output_file.write(columns.to_bytes(4, 'big'))
    for i in list_output:
        output_file.write(i.to_bytes(2, 'big'))
    output_file.close()

# Normalization using feature scaling between any arbitrary points 0 and 25500
def normalization(embedding):
    a = 0
    b = 25500
    min = np.min(embedding[0][0])
    max = np.max(embedding[0][0])
    # https://en.wikipedia.org/wiki/Feature_scaling
    normalized = [((b-a)*((i-min)/(max-min))+a).astype(int) for i in embedding[0][0]]
    normalized = np.concatenate(normalized).ravel().tolist()
    return normalized

# Write output file (2 bytes/pixel)
def write_outfile(pixels, numarray, autoencoder, imageset, outputname, parameters):
    if pixels is None:
        pixels, numarray = numpy_from_dataset(imageset, 4, False)
    newpixels = np.reshape(pixels, (-1, numarray[2], numarray[3]))
    embedding = list((K.function([autoencoder.input], [layer.output])(newpixels) for layer in autoencoder.layers if layer.output_shape == (None, parameters[5])))
    newlst = normalization(embedding)
    write_output(newlst, len(embedding[0][0]), outputname)
