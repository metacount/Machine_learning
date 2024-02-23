from google.cloud import bigquery
import pandas as pd
from typing import List, Any, Dict
import re

def slect_distinct(table_ids:List[str],
  project_id:str, dataset_id:str)->None:
  client = bigquery.Client(project=project_id)
  for table in table_ids:
      distinct_application_table = f'distinct_{table}'
      remove_duplicate = f"""
      CREATE TABLE `{project_id}.{dataset_id}.{distinct_application_table}` 
      AS
      SELECT DISTINCT
        *
      FROM 
      `{project_id}.{dataset_id}.{table}`;
      """
      query = client.query(remove_duplicate)



def dowlonad_tables(table_ids:List[str],project_id:str, 
                    dataset_id:str, drive_dir:str)->None:

  client = bigquery.Client(project=project_id)
  for table_id in table_ids:
      query = f'SELECT * FROM `{project_id}.{dataset_id}.{table_id}`'
      file_name = table_id + '.csv'
      df = client.query(query).to_dataframe()
      df.to_csv(drive_dir + file_name, index=False)

def upload_tables_bq(drive_dir:str, csv_files:List[str],
                    project_id:str, dataset_id:str, )->None:
    
    # creat table schema and uploads csv

    client = bigquery.Client(project=project_id)

    for csv_file in csv_files:
        csv_file_path = f'{drive_dir}{csv_file}'
        df = pd.read_csv(csv_file_path)
        pattern = re.compile(r'[^\w]')
        df.columns = [pattern.sub('', col_name) for col_name in df.columns]
        schema = []
        for col_name, dtype in df.dtypes.items():
          if 'object' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'STRING'))
          elif 'int' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'INTEGER'))
          elif 'float' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'FLOAT'))
          elif 'bool' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'BOOLEAN'))
          elif 'datetime' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'TIMESTAMP'))
          elif 'timedelta' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'TIMESTAMP'))
          elif 'category' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'STRING'))
          elif 'complex' in str(dtype):
              schema.append(bigquery.SchemaField(col_name, 'FLOAT'))
          else:
              print(f"Unrecognized dtype '{dtype}' for column '{col_name}'.\
               Mapping to STRING by default.")
              schema.append(bigquery.SchemaField(col_name, 'STRING'))

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            skip_leading_rows=1,
            source_format=bigquery.SourceFormat.CSV,
        )
        table_id = csv_file.replace(".csv", "")
        table_ref = f'{project_id}.{dataset_id}.{table_id}'
        job = client.load_table_from_dataframe(df, table_ref,
        job_config=job_config)
        del df
        job.result() 

