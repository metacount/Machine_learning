import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Any, Dict


def check_dup_cols(cols):
  print(len(cols))
  print(len(set(cols)))

def find_string(list,str):
  print([col for col in list if str in col])

def get_csv_files(csv_dir:str)->List:
  os.chdir(csv_dir)
  files = os.listdir()
  csv_files = [file for file in files if file.endswith('.csv')]
  return csv_files


def convert_to_bool(df: pd.DataFrame):
  columns = df.select_dtypes(include='number').columns.tolist()
  for column_name in columns:
    if df[column_name].nunique() == 2:
      df[column_name] = df[column_name].astype(bool)
  


def corr_features(corr_matrix:pd.DataFrame,threshold:int)->pd.DataFrame:
  num_features = corr_matrix.shape[0]
  correlated_features = []
  for i in range(num_features):
      for j in range(i+1, num_features):
          if corr_matrix.iloc[i, j] > threshold:
              correlated_features.append((corr_matrix.columns[i],
                                          corr_matrix.columns[j],
                                          corr_matrix.iloc[i, j]))
  df_correlated_features = pd.DataFrame(correlated_features, 
  columns=['Feature 1', 'Feature 2', 'Correlation'])
  return df_correlated_features


def to_csv(path:str,df:pd.DataFrame)->None:
  df.to_csv(path, index = False)


def corr_matrix(df:pd.DataFrame)-> pd.DataFrame:
  corr_matrix = df.corr().abs()
  return corr_matrix

def bin1(series: pd.Series, 
        low_range: int = None, high_range: int = None, 
        bins: int = 8)->pd.Series:

    if low_range is None or high_range is None:
        low_range = series.min() if low_range is None else low_range
        high_range = series.max() if high_range is None else high_range

    series = series[(series > low_range) & (series < high_range)]
    bins = pd.cut(series, bins=bins)
    bin_counts = bins.value_counts().sort_index()
    bin_counts.columns = [series.name, 'Count']
    return bin_counts

    
def bins(df:pd.DataFrame)->pd.DataFrame:
  for col in df.columns:
    print(bin1(df[col],col))


def value_counts(df:pd.DataFrame,columns:List):
  for column in columns:
    print(df[column].value_counts().sort_values())


def bin2(series1:pd.Series,series0:pd.Series,bins:int=8)-> pd.DataFrame:
  """
  bin two series, and see proportions of value_coutns
  """
  low = series1.min()
  high = series1.max()
  bin_edges = np.linspace(low, high, bins+1)
  bined1 = bin1(series1,low,high,bin_edges).reset_index()
  bined0 = bin1(series0,low,high,bin_edges).reset_index()
  result_df = pd.merge(bined0, bined1, 
                      on='index', how='outer', suffixes=('_n', '_p'))
  result_df['prop'] = (
      result_df[f'{series1.name}_p']) / result_df[f'{series0.name}_n']
  return result_df
