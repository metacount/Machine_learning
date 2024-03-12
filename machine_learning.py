import lightgbm as lgb
from scipy.optimize._lsq.common import evaluate_quadratic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pandas import DataFrame, Series
from typing import List, Tuple, Any, Dict
from lightgbm import Dataset, early_stopping, train
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
import joblib

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
  return df


def create_splits(df: DataFrame, target:str,test_size:int=0.3)->Tuple[
  DataFrame, DataFrame, Series, Series]:
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,stratify=y)
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


def feature_importance(clf: lgb.LGBMClassifier) -> pd.DataFrame:
  feature_importance = clf.feature_importances_
  feature_names = clf.feature_name_
  df_fi = pd.DataFrame({'Feature': feature_names,
                      'Importance': feature_importance})
  df_fi = df_fi.sort_values(by='Importance', ascending=False)
  return df_fi


def sanitize(df:pd.DataFrame)->pd.DataFrame:
    return df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
def lgbc(X_train: DataFrame, y_train: Series, 
          params: Dict[str, Any], 
          num_round: int = 10) -> Any:
  """
    Train and return a LightGBM model.
  """
  lgb_dataset = lgb.Dataset(data=X_train,
                            label=y_train,
                            )      
  clf = lgb.train(params, lgb_dataset, num_round)
  return clf

  
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



def train_val_split(df:pd.DataFrame,target):
  y = df[target]
  X = df.drop(columns=[target])
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=y)
  del X_test, y_test
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=y_train)
  return  X_train, y_train, X_val, y_val


def save_results(result_dict:Dict[str, Any],
                  name:str, clf_dir:str='classifiers')-> None:
  file_path = os.path.join(clf_dir, f"{name}_eval.json")
  result_dict['confusion_matrix'] = result_dict[
    'confusion_matrix'].tolist()
  with open(file_path, 'w') as f:
    json.dump(result_dict, f)

def get_results(result_dict_path:str):

  with open(result_dict_path, 'r') as f:
    result_dict = json.load(f)
  for key in result_dict.keys():
    print(key)
    print(result_dict[key])


def score_classifier(clf:lgb.LGBMClassifier, 
                     X:pd.DataFrame, 
                     y_true:pd.Series,
                     clf_name:str='default'
                     )->pd.DataFrame:
  file_path = 'csv/scores.csv'
  if os.path.exists(file_path):
    score_df = pd.read_csv(file_path)
  else:
    score_df = pd.DataFrame()
  new_index = len(score_df)
  score_df.loc[new_index, 'Classifier'] = clf_name
  preds_proba = roc_auc_score(y_true, clf.predict_proba(X)[:,1])
  score_df.loc[new_index, 'auc_proba'] = preds_proba
  y_pred = clf.predict(X)
  preds_binary = roc_auc_score(y_true,y_pred)
  score_df.loc[new_index, 'auc_binary'] = preds_binary
  score_df.to_csv('csv/scores.csv',index=False)


  eval_results = evaluate_clf(y_true,y_pred)
  save_results(eval_results,clf_name)
  joblib.dump(clf, f'classifiers/{clf_name}.pkl')
  return preds_proba


def lgb_classifier(df:DataFrame, 
                      params: Dict[str, Any],
                      target: str,
                      ) -> Tuple[lgb.LGBMClassifier, pd.DataFrame,
                      pd.Series]:

  X_train, y_train, X_val, y_val = train_val_split(df,target)
  clf = lgb.LGBMClassifier(**params)
  clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
  del X_train, y_train
  return clf, X_val, y_val


def classifier(df:pd.DataFrame, 
              params: Dict[str, Any],
              target: str,
              save_name:str = 'default'
              ):
  clf, X_val, y_val = lgb_classifier(df, params, target)
  score = score_classifier(clf, X_val, y_val, save_name)
  print(score)
  joblib.dump(clf, f'classifiers/{save_name}.pkl')
  return clf



def hyper_parameter_tuning(
  df:pd.DataFrame, 
  fixed_params: Dict[str, Any],
  search_space: Dict[str, Any],
  target: str,
  name:str
  )-> Dict[str, Any]:


  for key in search_space.keys():
    if key in fixed_params:
      del fixed_params[key]
      
  df = create_categorical(df)
  X_train, y_train, X_val, y_val = train_val_split(df, target)

  
  def objective_function(params):
    
      params['n_estimators'] = int(params['n_estimators'])
      params['num_leaves'] = int(params['num_leaves'])
      params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
      params['bagging_freq'] = int(params['bagging_freq'])

      combined_params = {**fixed_params, **params}
      clf = lgb.LGBMClassifier(**combined_params)
      clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
      
                
      y_pred = clf.predict_proba(X_val)[:,1]
      metric = roc_auc_score(y_val, y_pred)
     
      return -metric

  trials = Trials()
  best_params = fmin(fn=objective_function,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=50, trials=trials
                    )
  for key in best_params.keys():
    best_params[key]=int(best_params[key])
  com_params = {**best_params, **fixed_params}

  file_path = os.path.join('classifiers/', f"{name}.json")
  with open(file_path, 'w') as f:
    json.dump(com_params, f)
  return com_params

def final_classifier(df:pd.DataFrame,
                      clf:lgb.LGBMClassifier, 
                      target: str,
                      name:str,
                      ) ->Tuple[lgb.LGBMClassifier,float]:
  df = create_categorical(df)
  df = sanitize(df)
  y = df[target]
  X = df.drop(columns=[target])
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=y)

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=y_train)

  clf.fit(X_train, y_train,eval_set=[(X_val, y_val)])
  score = score_classifier(clf, X_test, y_test, name)


def submission_classifier(df:pd.DataFrame,
                      clf:lgb.LGBMClassifier, 
                      target: str,
                      name:str,
                      ) ->Tuple[lgb.LGBMClassifier,float]:
  df = create_categorical(df)
  df = sanitize(df)
  y = df[target]
  X = df.drop(columns=[target])
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=42,
                                                      stratify=y)
  clf.fit(X_train, y_train,eval_set=[(X_test, y_test)])
  score = score_classifier(clf, X_test, y_test, name)

  return score

                                                  












