from pandas import DataFrame

from typing import List, Tuple, Any, Dict
import numpy as np
import pandas as pd
import os
import re


def get_csv_files(root_dir:str, csv_dir:str)->List:
  os.chdir(root_dir + csv_dir)
  files = os.listdir()
  csv_files = [file for file in files if file.endswith('.csv')]
  os.chdir(root_dir)
  return csv_files

def sanitize(feature_name):
    sanitized_name = re.sub(r'[^\w]', '', feature_name)
    return sanitized_name

def get_numeric_columns(df:DataFrame)->List:
  return df.select_dtypes(include=np.number).columns

def aggregate_numerical(df: DataFrame,
                        functions:List,
                        id_column:str,
                        numeric_columns:List)->DataFrame:
  aggregations = {col: functions for col in numeric_columns}
  numeric_columns
  df_aggregated = df.groupby(id_column).agg(
              aggregations).reset_index()
  df_aggregated.columns = ['_'.join(col).strip()
                    for col in df_aggregated.columns.values]
  df_aggregated = df_aggregated.rename(
    columns={id_column + '_':id_column})
  return df_aggregated

def aggregate_categorical(df: DataFrame,
                        pid:str,
                        categorical_columns:List)->DataFrame:
  one_hot_encoded = pd.get_dummies(df[categorical_columns])
  df_encoded = pd.concat([df[pid],
  one_hot_encoded], axis=1)
  agg_dict = {col: 'sum' for col in one_hot_encoded.columns}
  df_encoded_agg = df_encoded.groupby(pid).agg(
    agg_dict).reset_index()
  return df_encoded_agg

def merge_target(df1: DataFrame,df2: DataFrame,pid, target) -> DataFrame:
  df_target = pd.merge(df1,df2[[pid,
                            target]], on=pid ,how='left')
  df_target.dropna(subset=['TARGET'], inplace=True)
  return df_target


def get_csv(root_dir, csv_dir, files_no_id):
  os.chdir(root_dir + csv_dir)
  files = os.listdir()
  csv_files = [file for file in files if file.endswith('.csv')]
  for file_to_remove in files_no_id:
      if file_to_remove in csv_files:
        csv_files.remove(file_to_remove)
  os.chdir(root_dir)
  return csv_files


def aggregate_tables(root_dir: str,
                     csv_files: List[str],
                     csv_dir: str,
                     agg_dir: str,
                     functions: List[str],
                     id_columns: List[str],
                     ) -> None:
    pid = id_columns[0]    
    os.chdir(root_dir + csv_dir)             
    for csv_file in csv_files:
      df = pd.read_csv(csv_file)
      numeric_columns = get_numeric_columns(df)
      existing_ids = [col for col in id_columns if col in df.columns]
      numeric_columns = numeric_columns.drop(existing_ids)
      categorical_columns = df.select_dtypes(include='object').columns
      os.chdir(root_dir + agg_dir)
      if not numeric_columns.empty:
        df_num_agg = aggregate_numerical(df, functions, pid, numeric_columns)
      if not categorical_columns.empty:
        df_cat_agg = aggregate_categorical(df,
         pid, categorical_columns)  
      if not numeric_columns.empty and not categorical_columns.empty:
          df_merge = pd.merge(df_num_agg, df_cat_agg, on=pid)
          df_merge.to_csv(f'aggregated_{csv_file}', index=False)
      elif not numeric_columns.empty:
          df_num_agg.to_csv(f'aggregated_{csv_file}', index=False)
      elif not categorical_columns.empty:
          df_cat_agg.to_csv(f'aggregated_{csv_file}', index=False)
    os.chdir(root_dir)


def add_ids(root_dir,csv_dir,id_file,no_id_files,pid,sid):
  id_csv = f'{root_dir}{csv_dir}{id_file}'
  df1 = pd.read_csv(id_csv, usecols=[pid,sid])
  for file_no_id in no_id_files:
    no_ids_agg =  f'{root_dir}{csv_dir}{file_no_id}'
    df2 = pd.read_csv(no_ids_agg)
    df_merged = pd.merge(df1, df2, on=sid,how='left')
    file_dir = f'{root_dir}{csv_dir}'
    df_merged.to_csv(f'{file_dir}ID_{file_no_id}',index=False)


def merge_dfs(root_dir:str,agg_dir:str, app_file: str,
               agg_csv_files: list, common_key: str) -> None:
    merged_df = pd.read_csv(app_file)
    for agg_file in agg_csv_files:
      agg_file =  f'{root_dir}{agg_dir}{agg_file}'
      df = pd.read_csv(os.path.join(root_dir, agg_file))
      merged_df = pd.merge(merged_df, df, on=common_key,how='left')
      del df
    merged_df.to_csv(f'{root_dir}csv/merged.csv',index=False)
    del merged_df





