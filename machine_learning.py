import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pandas import DataFrame, Series
from typing import List, Tuple, Any, Dict
from lightgbm import Dataset, train
import pandas as pd

from sklearn.model_selection import cross_val_score
from IPython.display import Image
from IPython.display import display
from graphviz import Graph, Digraph
import os
from sklearn.metrics import roc_auc_score
import json
import re
import pandas as pd
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, space_eval


def load_lgb_clasifier(path:str)->lgb.Booster:
   return lgb.Booster(model_file=path)

def create_categorical(df: DataFrame) -> Tuple[DataFrame, List[str]]:
  """
  Creates DataFrame with categorical datatype and a 
  list of categorical columns.
  """
  cat_cols = df.select_dtypes(include='object').columns
  df[cat_cols] = df[cat_cols].astype('category')
  final_cat_cols = df.select_dtypes(include='category').columns
  cat_cols_list = final_cat_cols.tolist()
  return df, cat_cols_list


def create_splits(df: DataFrame, target:str)->Tuple[
  DataFrame, DataFrame, Series, Series]:
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test
  



def create_binary(y_pred_prob:Series, threshold:int=0.5):
  y_pred_binary = (y_pred_prob > threshold).astype(int)
  return y_pred_binary

def evaluate_clf(y_test, y_pred):
  """
    Evaluate and print classification report and confusion matrix.
  """
  results_dict = {
    'classification_report': classification_report(y_test, y_pred),
    'confusion_matrix': confusion_matrix(y_test, y_pred),
    'roc_auc_score': roc_auc_score(y_test, y_pred)
  }
  return results_dict


def feature_importance(model: lgb.Booster,
                  columns: List[str]) -> pd.DataFrame:
  feature_importance = model.feature_importance(importance_type='gain')

  df_fi = pd.DataFrame({'Feature': columns,
                      'Importance': feature_importance})
  df_fi = df_fi.sort_values(by='Importance', ascending=False)
  return df_fi


def sanitize(df:pd.DataFrame)->pd.DataFrame:
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return df 
    
def lgbc(X_train: DataFrame, y_train: Series, params: Dict[str, Any],
              cat_cols: List[str] = None,
              num_round: int = 10, clf_dir: str = 'clasifiers') -> Any:
  """
    Train and return a LightGBM model.
  """
  lgb_dataset = lgb.Dataset(data=X_train,
                            label=y_train,
                            categorical_feature=cat_cols)      
  clf = lgb.train(params, lgb_dataset, num_round)
  return clf

def lgbc_from_df(df:DataFrame, params: Dict[str, Any],
              target: str, num_round: int = 10, 
              threshold:str =0.5,
              clf_dir: str = 'clasifiers') -> Any:
  
  df.columns = [sanitize(col_name) for col_name in df.columns]
  df, categorical_columns = create_categorical(df)
  X_train, X_test, y_train, y_test = create_splits(df, target)
  lgbc1 = lgbc(X_train, y_train, params, target,
                categorical_columns, num_round)
  y_pred = lgbc1.predict(X_test, num_iteration=lgbc1.best_iteration)
  y_pred_binary = create_binary(y_pred, threshold)
  eval_results = evaluate_clf(y_test,y_pred_binary)
  del X_train, X_test, y_train, y_test, df
  return lgbc1, eval_results

def save_results(clf:Any,result_dict:Dict[str, Any],
                  clf_dir:str,name:str)-> None:
  file_path = os.path.join(clf_dir, f"{name}.json")
  result_dict['confusion_matrix'] = result_dict[
    'confusion_matrix'].tolist()
  clf.save_model(clf_dir + f'/{name}.txt')
  with open(file_path, 'w') as f:
    json.dump(result_dict, f)

def get_results(result_dict_path:str):

  with open(result_dict_path, 'r') as f:
    result_dict = json.load(f)
  for key in result_dict.keys():
    print(key)
    print(result_dict[key])

  
def graph_tree(model: lgb.Booster, tree_indexes: List[str]) -> None: 
    image_filenames = []
    for index in tree_indexes:
      tree = lgb.create_tree_digraph(model, tree_index=index)
      tree.graph_attr.update({'bgcolor': '#2B2B2B', 'color': 'white', 
      'fontcolor': 'white'})
      tree.node_attr.update({'style': 'filled', 'fillcolor': '#666666',
       'fontcolor': 'white'})
      tree.edge_attr.update({'color': 'white'})
      filename = f'tree_{index}_diagram'
      tree.render(filename=filename, format='png', cleanup=True)
      image_filenames.append(f'{filename}.png')
    for file_name in image_filenames:
      display(Image(filename=file_name))


def select_features(high_corr_features:pd.DataFrame,
      important_features:pd.DataFrame)->pd.DataFrame:
  removed_features = set()

  for index, row in high_corr_features.iterrows():
      feat1_importance = important_features.loc[
          important_features['Feature'] == row['Feature 1'], 'Importance']
      feat2_importance = important_features.loc[
          important_features['Feature'] == row['Feature 2'], 'Importance']

      if not feat1_importance.empty and not feat2_importance.empty:
          feat1_importance = feat1_importance.values[0]
          feat2_importance = feat2_importance.values[0]
          if feat1_importance < feat2_importance:
              removed_features.add(row['Feature 1'])
          else:
              removed_features.add(row['Feature 2'])
  
  selected_features = important_features[
    ~important_features['Feature'].isin(removed_features)]
  return selected_features

def selected_features(corr_features:pd.DataFrame,
                    feature_importance:pd.DataFrame,
                    corr_treshold:int,
                    importance_threshold:int,
                    )->List:

  high_corr_features = corr_features[
    corr_features['Correlation']>corr_treshold]
  important_features = feature_importance[
    feature_importance['Importance']>importance_threshold]
  selected_features = select_features(high_corr_features,important_features)
  selected_columns = selected_features['Feature'].tolist()
  return selected_columns


def lgb_hp_tuning(X_train:pd.DataFrame,
                  y_train:pd.Series,
                  space:Dict[str,Any],
                  num_round:int,
                  early_stopping_rounds:int,
                  scale_pos_weight:int,
                  nfold:int,
                  max_evals:int):

  cat_cols = X_train.select_dtypes(include=['category']).columns.tolist()
  lgb_dataset = lgb.Dataset(data=X_train, label=y_train,
                            categorical_feature=cat_cols,
                            free_raw_data=False)

  def objective(params):

      lgb_params = {
          'objective': 'binary',
          'metric': 'roc_auc',
          'verbosity': -1,
          'early_stopping_rounds':early_stopping_rounds,
          'scale_pos_weight': scale_pos_weight,
          'force_row_wise': True,
          **params
      }
      cv_results = lgb.cv(params=lgb_params,
                          train_set=lgb_dataset,
                          num_boost_round=num_round,
                          nfold=nfold,
                          seed=42,
                          metrics='auc'
                          )
      mean_auc = np.mean(cv_results['valid auc-mean'])
      return -mean_auc
  trials = Trials()
  best = fmin(fn=objective,
              space=space,
              algo=tpe.suggest,
              max_evals=max_evals,
              trials=trials)
  best_params = space_eval(space, best)
  return best_params













