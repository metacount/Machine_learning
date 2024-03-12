import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Any, Dict
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from scipy.stats import iqr
import scipy.stats as stats

plt.style.use("dark_background")
pio.templates["custom_dark"] = pio.templates["plotly_dark"]

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
  


def corr_features(corr_matrix:pd.DataFrame,threshold:int=0.7
                  )->pd.DataFrame:
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

    series = series[(series >= low_range) & (series <= high_range)]
    bins = pd.cut(series, bins=bins)
    bin_counts = bins.value_counts().sort_index()
  

    return bin_counts

    
def bins(df:pd.DataFrame)->pd.DataFrame:
  for col in df.columns:
    print('')
    print(f'{bin1(df[col])}')
    print('')


def value_counts(df:pd.DataFrame):
  for column in df.columns:
    print('')
    print(f'{df[column].value_counts().sort_values()}')
    print('')

def view_columns(df:pd.DataFrame):
  convert_to_bool(df)
  num_df = df.select_dtypes(include='number')
  cat_df = df.select_dtypes(exclude='number')
  print('-------NUMERICAL-----')
  bins(num_df)
  print('-------CATEGORICAL-----')
  value_counts(cat_df)

def view_dfs(path, csv_files:List,columns:List=None):

  for csv_file in csv_files:
    full_path = path + csv_file
    print(csv_file[:-4])
    print('------------------')
    print('')
    df =  pd.read_csv(full_path)
    if columns:
      existing_cols = [col for col in columns if col in df.columns]
      df = df.drop(existing_cols, axis=1)
      view_columns(df)
    else:
      view_columns(df)
  del df


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

def plot(df:pd.DataFrame, col:str, width:int=8, 
          height:int=2, mult:int=10e+3):
    get_left_value = lambda x: x[0].left.round(1)

    df['left_values'] = pd.Series([get_left_value(row) for index, row in df.iterrows()])

    x =df['prop']*mult
    plt.figure(figsize=(width, height))
    plt.plot(df[f'{col}_p'], marker='x', label=f'{col}_p', color='red')
    plt.plot(df[f'{col}_n'], marker='o', label=f'{col}_n',color='green')
    plt.plot(x, marker='v', label='prop',color='yellow')  
  
    plt.title(f'{col}')   
    plt.xticks(range(len(df['index'])), df['left_values'], rotation=45)
    plt.legend()
    plt.show()

def plot_distribution(cat_df:pd.DataFrame,column:str,target:str):
  df_grouped = cat_df.groupby([column,
                              target]).size().unstack(fill_value=0)

  df_grouped = df_grouped.head(20)
  df_grouped.plot(kind="bar", stacked=False, color=["green", "red"])
  plt.ylabel("Count")
  plt.title(column)
  plt.legend(title="Target")
  plt.tight_layout()
  plt.show()

def plot_proportion_in_one_row(df:pd.DataFrame, 
                            norm:str, target:str='TARGET'):
    fig, axes = plt.subplots(1, len(df.columns[:-1]), figsize=(16, 8))
    columns = df.columns.tolist()
    columns.remove(target)
    for i, col in enumerate(columns):
  
        proportion_table = (pd.crosstab(df[col], df[target], 
                                        normalize=norm) * 100).round(2)
        ax = axes[i]
        sns.heatmap(proportion_table, annot=True, cmap='crest',
                    cbar=False, ax=ax)
        ax.set_title(f" {col}")

    plt.tight_layout()
    plt.show()

def plot_columns(cat_df:pd.DataFrame,target:str):

  
  plt.style.use("dark_background")
  dfpc = cat_df[cat_df[target]==True].copy()
  dfnc = cat_df[cat_df[target]==False].copy()
  nunique_values = cat_df.nunique()
  columns_to_hist = nunique_values[nunique_values >= 20].index.tolist()
  columns_to_prop = nunique_values[nunique_values < 20].index.tolist()

  for column in columns_to_hist:
    plot_distribution(cat_df,column, target)
  plot_proportion_in_one_row(cat_df[columns_to_prop],'all')
  plot_proportion_in_one_row(cat_df[columns_to_prop],'columns')
  plot_proportion_in_one_row(cat_df[columns_to_prop],'index')


def plot2d(df_plot, columns,height,width, log=True):
  if log == True:
    size = 'log_size'
  else:
    size = 'size'
  
  fig = px.scatter(df_plot, x=columns[0], y=columns[1],
                   size=size, 
                  color='mean', opacity=0.7,
                 labels={columns[0]: columns[0], columns[1]: columns[1]})
  fig.update_layout(title='Target mean and log size', 
  height=height, width=width)
  fig.layout.template = 'custom_dark'
  return(fig.show())

def plot3d(df_plot, columns):
  fig = px.scatter_3d(df_plot, x=columns[0], y=columns[1], z=columns[2], 
  color='mean', size = 'log_size', opacity=0.7)
  fig.update_layout(scene=dict(xaxis_title=columns[0], yaxis_title=columns[1], 
  zaxis_title=columns[2]),
                    title='3D Target mean and log size',
                    height=800, width=1200)
  fig.layout.template = 'custom_dark'
  return(fig.show())

def mean_and_size(df, columns, target):
  df_agg = df.groupby(columns,observed=True)[target].agg([
    'size', 'mean']).reset_index()
  df_agg['log_size'] = np.log(df_agg['size'])
  return df_agg


def num_eda(df,target,p_scale=10e+1, bins=8):
  outliers_iqr = pd.DataFrame(columns=df.columns)

  for column in df.drop(target, axis=1).columns:

    q1 = df[column].quantile(0.05)
    q3 = df[column].quantile(0.95)
    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)
    lower_fence = q1 - (1.5 * iqr)
    cols = [column, 'TARGET']

    if df[column].nunique() > 4:
      outliers_h_iqr = df[df[column] > upper_fence][cols]
      df_out_h = df[cols].loc[outliers_h_iqr.index]
      outliers_l_iqr = df[df[column] < lower_fence][cols]
      df_out_l = df[cols].loc[outliers_l_iqr.index]

      outliers_iqr = df[ (df[column] > upper_fence) | 
      (df[column] < lower_fence) ][cols]
      

      df_in = df[~df.index.isin(outliers_iqr.index)]

      print(column)

      dfplth = num_prop(df_out_h, column, target,'high',bins)
      dfpltl = num_prop(df_out_l, column, target,'low',bins)
      dfplt = num_prop(df_in, column, target,'middle',bins)

      if df_in[column].nunique() > 4:
        plot(dfplt,column,mult=p_scale)
      if not dfpltl.empty:
        print('lower')
        plot(dfpltl,column,mult=p_scale)
      if not dfplth.empty:
        print('higher')
        plot(dfplth,column,mult=p_scale)

      print('middle', df_in[column].nunique())
      print('higher', df_out_h[column].nunique())
      print('lower', df_out_l[column].nunique())

    
    else:
      print('-------------------------------------------\n\n\n')
      print(column + ', has few values in df')
      print(df[column].nunique())
    print('-------------------------------------------\n\n\n')
def add_target(df, app_t, pid, target):
  target_df = df.merge(app_t[[pid,target]], on = pid)
  return target_df

import pandas as pd
import scipy.stats as stats



def num_prop(df,column, target, position, bins):

  cols = [column, 'TARGET']
  dfp = df[cols][df[target]==1].copy()
  dfn = df[cols][df[target]==0].copy()
  if (not dfp.empty) & (not dfn.empty):
    if (dfp[column].nunique() > 4) & (dfn[column].nunique() > 4):
      dfplt = bin2(dfp[column],dfn[column],bins)
      dfplt = dfplt.sort_values(by ='index').reset_index().drop(
                                              columns=['level_0'])
    else:
      dfplt = pd.DataFrame()

  else:
    dfplt = pd.DataFrame()
  
  return dfplt

def remove_nan(df):
  nan_counts = df.isna().sum()
  no_nan = df.drop(nan_counts[nan_counts > 1000].index, axis=1)
  no_nan = no_nan.dropna()
  return no_nan

def plot_round(df,columns, target,log = False, height=400,width=700):
  
  for column in columns:
    df[column] = df[column].round(-3)
    print(df[column].nunique())
    if df[column].nunique() > 100:
      df[column] = df[column].round(-4)
      print(df[column].nunique())
      if df[column].nunique() > 50:
        df[column] = df[column].round(-5)
        print(df[column].nunique())
  mean_size = mean_and_size(df, columns, target)
  if len(columns)<3:
    plot2d(mean_size,columns,height=height,width=width,log = log)
  else:
    plot3d(mean_size,columns)

def drop_outliers(df,columns, thresh=3):
  filtered_df = df.copy()
  for col in columns:


    q1 = filtered_df[col].quantile(0.25)
    q3 = filtered_df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (thresh * iqr)
    upper_bound = q3 + (thresh * iqr)
    filtered_df = filtered_df[ (filtered_df[col] >= lower_bound) &
                              (filtered_df[col] <= upper_bound)]
  return filtered_df




