from torch import nn

import pandas as pd
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device: {}".format(device))


class Generator(nn.Module):

    def __init__(self, z_size=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_size, 128)
        self.nl1 = nn.LeakyReLU(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.nl2 = nn.LeakyReLU(0.125)
        self.fc3 = nn.Linear(64, 4)
    
    def forward(self, x):
        o = self.nl1(self.fc1(x))
        o = self.nl2(self.fc2(o))
        return self.fc3(o)
        

if __name__ == "__main__":
    
    G = Generator().to(device)
    G.load_state_dict(torch.load("model.pt"))
    G.to(device)

    train_df = pd.read_csv("train.csv")
    noise_df = pd.read_csv("noise.csv")
    
    noise = torch.tensor(noise_df.values).to(device).float()
    output = G(noise).cpu().detach().numpy()

    output_df = pd.DataFrame(output, columns=train_df.columns)
    # need un-normalization since the generator has been trained on normalized data
    output_df = (output_df * train_df.std()) + train_df.mean()
    output_df.to_csv("outputs.csv", index=None)