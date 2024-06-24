# Databricks notebook source

def create_vector_search_if_not_exists(vector_search_endpoint_name: str, client) -> bool:
  if(client.list_endpoints()['endpoints']):
    if vector_search_endpoint_name not in [item['name'] for item in client.list_endpoints()['endpoints']]:
      print("Creating new VSE " + vector_search_endpoint_name)
      client.create_endpoint(
        name= vector_search_endpoint_name,
        endpoint_type="STANDARD"
      )
      return True
    else: 
      print("Vector search endpoint: " + vector_search_endpoint_name + " already exists!")
      return False
  else:
    print("Creating new VSE " + vector_search_endpoint_name)
    client.create_endpoint(
      name= vector_search_endpoint_name,
      endpoint_type="STANDARD"
    )
    return True


# COMMAND ----------


def create_vector_index_if_not_exists(vector_search_endpoint_name: str, vector_index_name:str, uc_save_path:str, embeddings_model:str, client) -> bool:

  if(client.list_indexes(vector_search_endpoint_name)['vector_indexes']):
    if vector_index_name not in [item['name'] for item in client.list_indexes(vector_search_endpoint_name)['vector_indexes']]:
      print("Creating vector index: " + vector_index_name)
      index = client.create_delta_sync_index(
      endpoint_name= vector_search_endpoint_name,
      source_table_name= f"{uc_save_path}.raw_mlflow_docs",
      index_name= vector_index_name,
      pipeline_type='TRIGGERED',
      primary_key="id",
      embedding_source_column= "page_content",
      embedding_model_endpoint_name= embeddings_model
    )
      return True
    else: 
      print("Vector index: " + vector_index_name + " already exists!")
      return False
  else:
    print("Creating vector index: " + vector_index_name)
    index = client.create_delta_sync_index(
    endpoint_name= vector_search_endpoint_name,
    source_table_name= f"{uc_save_path}.raw_mlflow_docs",
    index_name= vector_index_name,
    pipeline_type='TRIGGERED',
    primary_key="id",
    embedding_source_column= "page_content",
    embedding_model_endpoint_name= embeddings_model
    )
    return True


# COMMAND ----------


