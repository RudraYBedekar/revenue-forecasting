import pandas as pd

def preprocess_data(df):

    df = df.fillna(0)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week

    df["IsHoliday"] = df["IsHoliday_x"].astype(int)
    df.drop(columns=["IsHoliday_x","IsHoliday_y"], errors="ignore", inplace=True)


    return df