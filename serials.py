import torch
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from data import DataLoader as DL
from Decoder3 import Decoder as DE
from Regressor import MultiOutputRegressor as MRe
import random

ae_layers = 3
latent_dim = 60
file_length = 1633
train_ratio = 0.8
spectra_dim = 1201
geo_numbers = 6
hid_dim = 130
hid_dim1 = hid_dim
hid_dim2 = hid_dim
hid_dim3 = hid_dim
hid_dim4 = hid_dim
hid_dim5 = hid_dim
ANN_train_epoch = 10000
ANN_weight = 'latent_ANN' + str(hid_dim1) + '_weight.pth'
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DE = DE(input_dim=spectra_dim, latent_dim=latent_dim)
Geo2Latent = MRe(input_dim=geo_numbers, output_dim=latent_dim, hid1=hid_dim1,
                 hid2=hid_dim2, hid3=hid_dim3, hid4=hid_dim4, hid5=hid_dim5)


def generate_random_list(start, end):
    num_list = list(range(start, end + 1))
    random.shuffle(num_list)
    return num_list


def xlsx2csv(xlsx_file, csv_file):
    this_data = pd.read_excel(xlsx_file, header=None)
    this_data.to_csv(csv_file, header=False, index=False)


def contact_xlsx(xlsx_file1, xlsx_file2, csv_file):
    file1 = pd.read_excel(xlsx_file1, header=None)
    file2 = pd.read_excel(xlsx_file2, header=None)
    if file1.shape[0] != file2.shape[0]:
        raise ValueError("COLUMN DIS MATCHED\n")
    combined_file = pd.concat([file1, file2], axis=1)
    combined_file.to_csv(csv_file, header=False, index=False)


random_list = generate_random_list(0, 1632)
Train_idx = random_list[:round(1633 * 0.8)]
Test_idx = random_list[round(1633 * 0.8):]
spectra = DL('./dataset/final_gfactor.CSV')
Spectra = spectra.load_data()
Xall = Spectra.iloc[:, :geo_numbers]
Yall = Spectra.iloc[:, geo_numbers:]
Yall = torch.tensor(Yall.values, dtype=torch.float32)
Xall = torch.tensor(Xall.values, dtype=torch.float32)
ANN_Train_idx = random_list[:round(file_length * train_ratio)]
ANN_Test_idx = random_list[round(file_length * train_ratio):]
ANN_dataset = TensorDataset(Xall, Yall)
ANN_train_set = Subset(ANN_dataset, ANN_Train_idx)
ANN_test_set = Subset(ANN_dataset, ANN_Test_idx)
ANN_train_loader = DataLoader(ANN_train_set, batch_size=256, shuffle=True, num_workers=0)
ANN_test_loader = DataLoader(ANN_test_set, batch_size=1024, shuffle=False, num_workers=0)
ANN_all_loader = DataLoader(ANN_dataset, batch_size=2048, shuffle=False, num_workers=0)
ANN_optimizer = torch.optim.Adam(Geo2Latent.parameters(), lr=1e-4)
criterion = nn.MSELoss()
Geo2Latent.to(DEVICE)
Geo2Latent.train()
weight = torch.load('./weight/l1/AE60weight.pth')
DE.load_state_dict(weight)
DE.to(DEVICE)
DE.eval()
num_epochs = ANN_train_epoch
for epoch in range(num_epochs):
    for X, Y in ANN_train_loader:
        Geo2Latent.train()
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        la = Geo2Latent(X)
        outputs = DE(la)
        loss = criterion(outputs, Y)
        ANN_optimizer.zero_grad()
        loss.backward()
        ANN_optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'ANN Training... : Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}')
        with torch.no_grad():
            for X, Y in ANN_test_loader:
                Geo2Latent.eval()
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                la = Geo2Latent(X)
                outputs = DE(la)
                loss = criterion(outputs, Y)
                print(f'ANN Training... : Epoch [{epoch + 1}/{num_epochs}], Test Loss: {loss.item():.6f}')
                file_str = f"./weight/ann/model{epoch}"
                torch.save(Geo2Latent.state_dict(), file_str)
ann_name = './weight/w' + str(hid_dim) + 'ANN_weight.pth'
torch.save(Geo2Latent.state_dict(), ann_name)
print("ANN training finished")
name = './ann/dim' + str(hid_dim) + 'recon.xlsx'
Geo2Latent.eval()
for X in ANN_all_loader:
    X = X[0].to(DEVICE)
    la = Geo2Latent(X)
    outputs = DE(la)
    detached = outputs.detach()
    cpu_tensor = detached.cpu()
    numpy_array = cpu_tensor.numpy()
    df = pd.DataFrame(numpy_array)
    df.to_excel(name, index=False, header=False)

