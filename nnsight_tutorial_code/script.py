from collections import OrderedDict

import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)

# random input
input = torch.rand((1, input_size))

from nnsight import NNsight

model = NNsight(net)

print(model)

input = torch.rand((1, input_size))

with model.trace(input):
    # Your intervention code goes here
    # The model runs when the context exits
    pass
