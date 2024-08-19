from scnn.bipolar_functions.layers import *
from scnn.bipolar_functions.operations import *
from scnn.networks.network1 import *
import numpy as np
import torch
import tensorflow as tf

def get_norm_weight(weight,bias):
    compact = torch.concat([weight,bias.unsqueeze(0)],dim=0)
    scalar = torch.max(torch.abs(compact))
    scaled_weight = compact / scalar
    return scalar, scaled_weight[:-1,:], scaled_weight[-1]

def get_weight_from_raw(file_path):
    data_raw = np.load(file_path)

    data_raw = {x:torch.tensor(data_raw[x]) for x in data_raw}

    s1, w1, b1 = get_norm_weight(data_raw['arr_0'],data_raw['arr_1'])
    s2, w2, b2 = get_norm_weight(data_raw['arr_2'],data_raw['arr_3'])

    data = {
        'layer1':{
            'weight':w1,
            'bias':b1,
            'nmk':s1
        },
        'layer2':{
            'weight':w2,
            'bias':b2,
            'nmk':s2,
        },
    }
    return data



if __name__ == '__main__':
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    x_test = np.array((x_test)/255.0,dtype="<f4")
    
    x_test, y_test = torch.tensor(x_test)[0:1000] , torch.tensor(y_test)[0:1000]

    seq_len=16 #1024


    file_path = '/Users/ian/Desktop/github/sc/adiabaticbinary/normal_training/Dse__epoc14.0.npz'
    weight_data = get_weight_from_raw(file_path)

    f2s = F2S(seq_len=seq_len)

    model = model_axpc(seq_len=seq_len,data=weight_data)

    total = len(y_test)
    count = 0
    for i in range(total):
        out = model.forward(f2s(x_test[i]))
        pred = out.max(dim=-1)[1]
        if pred == y_test[i]:
            count += 1
        print(f"{i+1}/{total}: acc:{count/(i+1)}")