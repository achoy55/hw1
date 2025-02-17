import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


def normalize_MinMaxScaler(X_train, X_val, X_test):
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def normalize_MinMax_by_column(X_train, y_train):
    sc = MinMaxScaler()
    X_df, y_df = sc.fit_transform(np.column_stack((X_train, y_train))).T
    return X_df, y_df

def normalize_StandardScaler(X_train, X_val, X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def normalize_RobustScaler(X_train, X_val, X_test):
    sc = RobustScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def normalize_MaxAbsScaler(X_train, X_val, X_test):
    sc = MaxAbsScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    X_test_scaled = sc.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled
