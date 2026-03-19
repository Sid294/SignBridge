import pandas as pd
import torch

def load_data(train=True):
    file_name = "data/sign_mnist_train.csv" if train else "data/sign_mnist_test.csv"
    df = pd.read_csv(file_name)

    X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32) / 255.0
    y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

    X = X.view(-1, 1, 28, 28)  # reshape for CNN
    return X, y