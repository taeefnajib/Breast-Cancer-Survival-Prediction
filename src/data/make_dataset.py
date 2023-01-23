import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

ROOTDIR = os.getcwd()

# Getting the dataset
def create_dataframe(filepath):
    return pd.read_csv(filepath)


# Preparing the dataset
def prepare_dataset(df):
    cat_cols = ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "Grade"]
    df.rename(columns={"T Stage ": "T Stage"}, inplace=True)
    df["differentiate"].replace({"Moderately differentiated": 2,
                                "Poorly differentiated": 1,
                                "Well differentiated": 3,
                                "Undifferentiated": 0}, inplace=True)
    df["A Stage"].replace({"Regional":1, "Distant": 0}, inplace=True)
    df["Estrogen Status"].replace({"Positive":1, "Negative": 0}, inplace=True)
    df["Progesterone Status"].replace({"Positive":1, "Negative": 0}, inplace=True)
    df["Status"].replace({"Alive":1, "Dead": 0}, inplace=True)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df.to_csv(f"{ROOTDIR}/data/processed/processed_patient_data.csv")
    return df

# Scaling features
def scale_features(X):
    scaler = StandardScaler()
    cols = X.columns
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=cols)

# Creating feature and target dataframe
def create_feat_target(df):
    X = df.drop(["Status"], axis=1)
    y = df["Status"]
    X = scale_features(X=X)
    return X, y

# Creating train and validation datasets
def create_train_val(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Making final features (X) and target (y) for model training
def make_dataset(filepath):
    df = create_dataframe(filepath=filepath)
    df = prepare_dataset(df)
    return create_feat_target(df)