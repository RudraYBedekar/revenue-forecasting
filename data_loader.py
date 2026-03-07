import pandas as pd

def load_data():
    train = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")
    stores = pd.read_csv("data/stores.csv")

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    return train, features, stores


def merge_datasets(train, features, stores):
    df = train.merge(features, on=["Store", "Date"], how="left")
    df = df.merge(stores, on="Store", how="left")

    return df