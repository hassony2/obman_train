from torch import nn


class AbsoluteBranch(nn.Module):
    def __init__(self, base_neurons=[515, 256], out_dim=3):
        super().__init__()

        layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            zip(base_neurons[:-1], base_neurons[1:])
        ):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out
