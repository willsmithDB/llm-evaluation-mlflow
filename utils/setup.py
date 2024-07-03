# Databricks notebook source
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
volume_name = dbutils.widgets.get("volume_name")
vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")

# COMMAND ----------

dbutils.widgets.text("volume_path", f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/")
dbutils.widgets.text("uc_save_path", f"{catalog_name}.{schema_name}")

# which embeddings model we want to use. We are going to use the foundation model API, but you can use custom models (i.e. from HuggingFace), External Models (Azure OpenAI), etc.
dbutils.widgets.text("embeddings_model", "databricks-bge-large-en")

# Vector Index Name 
dbutils.widgets.text("vector_index", f"{catalog_name}.{schema_name}_vse.{volume_name}_embeddings")

# databricks-meta-llama-3-70b-instruct or databricks-dbrx-instruct
dbutils.widgets.dropdown("llm_model", choices = ["databricks-dbrx-instruct", " databricks-meta-llama-3-70b-instruct"], defaultValue = "databricks-dbrx-instruct")

# Target VSE Schema Name
dbutils.widgets.text("vse_schema_name", f"{schema_name}_vse")

# COMMAND ----------

#get widget values

volume_path = dbutils.widgets.get("volume_path")
uc_save_path = dbutils.widgets.get("uc_save_path")

llm_model = dbutils.widgets.get("llm_model")
embeddings_model = dbutils.widgets.get("embeddings_model")
vector_index_name = dbutils.widgets.get("vector_index")
vector_index_schema = dbutils.widgets.get("vse_schema_name")
