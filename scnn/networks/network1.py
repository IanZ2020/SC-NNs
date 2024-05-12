from ..bipolar_functions.layers import *
from ..bipolar_functions.operations import *

class model():
    def __init__(self, seq_len):
        self.in_features = 784
        self.out_features = 10

        self.layer1 = APCLinearAct(
            in_features=784,
            out_features=128,
            seq_len=seq_len,
            btanh_scalar=1,
            num_au_layers=1,
            is_bias=True,
        )


        self.layer2 = APCLinear(
            in_features=128,
            out_features=10,
            seq_len=seq_len,
            num_au_layers=1,
        )

    def load_weight(self, data):
        self.layer1.load_weight(data['layer1'])
        self.layer2.load_weight(data['layer2'])


    def forward(self, inputs):
        out = self.layer1.forward(inputs.flatten(-3,-2))
        out = self.layer2.forward(out)
        return torch.softmax(out,dim=-1)