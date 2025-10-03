import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET = 'koi_disposition'

DISPOSITION_MAP = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['rowid','kepid','kepoi_name','kepler_name','koi_comment','koi_disp_prov','koi_vet_date']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    df = df.dropna(axis=1, how='all')

    datalink_mask = df.columns.str.contains('datalink') | df.columns.str.contains('koi_datalink')
    df = df.loc[:, ~datalink_mask]

    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].str.strip()

    return df

def prepare_feature_label(df: pd.DataFrame):
    df = basic_clean(df)

    if df[TARGET].dtype == 'O':
        df[TARGET] = df[TARGET].str.upper().str.strip()
    else:
        try:
            df[TARGET] = df[TARGET].map({0:'CANDIDATE',1:'CONFIRMED',2:'FALSE POSITIVE'})
        except Exception:
            pass
    feature_cols = [c for c in df.columns if c != TARGET]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    return X, y

def build_pipeline(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ], remainder='drop')

    return preprocessor
