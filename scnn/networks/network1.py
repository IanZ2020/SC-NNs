from ..bipolar_functions.layers import *
from ..bipolar_functions.operations import *
from ..convertors import *

class model_axpc():
    def __init__(self, seq_len, data):
        self.seq_len = seq_len
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
            is_bias=True,
        )

        self.load_weight(data)

    def load_weight(self, data):
        self.layer1.load_weight(data['layer1'])
        self.layer2.load_weight(data['layer2'])


    def forward(self, inputs):
        s2f = S2F()
        out = self.layer1.forward(inputs.flatten(-3,-2))
        out = self.layer2.forward(out)
        return torch.softmax(out,dim=-1)

class model_apc():
    def __init__(self, seq_len, data):
        self.seq_len = seq_len
        self.in_features = 784
        self.out_features = 10
        
        self.nmk1 = data['layer1']['nmk']
        self.nmk2 = data['layer2']['nmk']

        self.layer1 = APCLinearAct(
            in_features=784,
            out_features=128,
            seq_len=seq_len,
            scalar=self.nmk1,
            num_au_layers=0,
            is_bias=True,
        )

        self.layer2 = APCLinear(
            in_features=128,
            out_features=10,
            seq_len=seq_len,
            scalar=self.nmk2,
            num_au_layers=0,
            is_bias=True,
        )

        self.load_weight(data)

    def load_weight(self, data):
        self.layer1.load_weight(data['layer1'])
        self.layer2.load_weight(data['layer2'])


    def forward(self, inputs):
        s2f = S2F()
        out = self.layer1.forward(inputs.flatten(-3,-2))
        out = self.layer2.forward(out)
        return torch.softmax(out,dim=-1)

class model_mux():
    def __init__(self, seq_len, data):
        self.seq_len = seq_len
        self.in_features = 784
        self.out_features = 10
        
        self.nmk1 = data['layer1']['nmk']
        self.nmk2 = data['layer2']['nmk']

        self.layer1 = MUXScaled_LinearAct(
            scalar=self.nmk1,
            in_features=784,
            out_features=128,
            seq_len=seq_len,
            is_bias=True,
        )

        self.layer2 = MUXScaled_Linear(
            in_features=128,
            out_features=10,
            seq_len=seq_len,
            is_bias=True,
        )

        self.load_weight(data)

    def load_weight(self, data):
        self.layer1.load_weight(data['layer1'])
        self.layer2.load_weight(data['layer2'])


    def forward(self, inputs):
        s2f = S2F()
        out = self.layer1.forward(inputs.flatten(-3,-2))
        out = self.layer2.forward(out)
        return torch.softmax(out,dim=-1)

class model_fp():
    def __init__(self, data):
        self.in_features = 784
        self.out_features = 10
        
        self.nmk1 = data['layer1']['nmk']
        self.nmk2 = data['layer2']['nmk']

        self.layer1 = torch.nn.Linear(
            in_features=784,
            out_features=128,
            bias=True,
        )
        self.layer2 = torch.nn.Linear(
            in_features=128,
            out_features=10,
            bias=True,
        )
        self.load_weight(data)

    def load_weight(self, data):
        self.layer1.weight =torch.nn.Parameter( data['layer1']['weight'].T)
        self.layer1.bias = torch.nn.Parameter(data['layer1']['bias'])
        self.layer2.weight = torch.nn.Parameter(data['layer2']['weight'].T)
        self.layer2.bias = torch.nn.Parameter(data['layer2']['bias'])

    def forward(self, inputs):
        out = torch.tanh(self.layer1(inputs.flatten(-2,-1)) * self.nmk1)
        out = self.layer2(out)*self.nmk2
        return torch.softmax(out,dim=-1)