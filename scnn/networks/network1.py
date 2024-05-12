from ..bipolar_functions.layers import *
from ..bipolar_functions.operations import *

class model():
    def __init__(self, seq_len, data):
        self.in_features = 784
        self.out_features = 10
        
        self.nmk1 = data['layer1']['nmk']
        self.nmk2 = data['layer2']['nmk']

        self.layer1 = APCLinearAct(
            in_features=784,
            out_features=128,
            seq_len=seq_len,
            scalar=self.nmk1,
            num_au_layers=1,
            is_bias=True,
        )


        self.layer2 = APCLinear(
            in_features=128,
            out_features=10,
            seq_len=seq_len,
            scalar=self.nmk2,
            num_au_layers=1,
        )

        self.load_weight(data)

    def load_weight(self, data):
        self.layer1.load_weight(data['layer1'])
        self.layer2.load_weight(data['layer2'])


    def forward(self, inputs):
        out = self.layer1.forward(inputs.flatten(-3,-2))
        out = self.layer2.forward(out)
        return torch.softmax(out,dim=-1)