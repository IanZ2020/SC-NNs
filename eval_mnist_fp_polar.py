from scnn.bipolar_functions.layers import *
from scnn.bipolar_functions.operations import *
from scnn.networks.network1 import model_1, model_1_fp
import numpy as np
import torch
import tensorflow as tf

def get_real_weight(weight, kk):
    return torch.tanh(weight*kk)

def get_weight_from_raw(file_path):
    data_raw = np.load(file_path)
    data_raw = {x:torch.tensor(data_raw[x]) for x in data_raw}
    kk = data_raw['arr_3']

    data = {
        'layer1':{
            'weight':get_real_weight(data_raw['arr_0'],kk),
            'bias':get_real_weight(data_raw['arr_1'],kk),
            'nmk':data_raw['arr_2']
        },
        'layer2':{
            'weight':get_real_weight(data_raw['arr_4'],kk),
            'bias':get_real_weight(data_raw['arr_5'],kk),
            'nmk':data_raw['arr_6']
        },
    }
    return data



if __name__ == '__main__':
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    x_test = np.array((x_test)/255.0,dtype="<f4")
    x_test, y_test = torch.tensor(x_test)[0:1000] , torch.tensor(y_test)[0:1000]



    file_path = '/Users/ian/Desktop/github/sc/adiabaticbinary/binary_97.0/Dse__w20.0_epoc14.0.npz'
    weight_data = get_weight_from_raw(file_path)


    model = model_1_fp(data=weight_data)

    total = len(y_test)
    count = 0
    for i in range(total):
        out = model.forward(x_test[i])
        pred = out.max(dim=-1)[1]
        if pred == y_test[i]:
            count += 1
        print(f"{i+1}/{total}: acc:{count/(i+1)}")