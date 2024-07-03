# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation of RAG (Retrieval-Augmented Generation) chain using Databricks Model Serving (Llama3-8b) and MLflow!
# MAGIC
# MAGIC - We will use langchain to pull MLflow documentation and chunk it. 
# MAGIC - We will pull meta-llama/Meta-Llama-3-8B-Instruct and deploy to Databricks Model Serving. 
# MAGIC - We will use the Databricks Model Serving endpoint to automatically compute embeddings from the chunks. 
# MAGIC - We will then create an index within a Databricks Vector Search index to hold the embeddings and act as a retriever for our RAG chain. 
# MAGIC - We log all of this in mlflow so that we can have the run history and associated artifacts stored!
# MAGIC - After creating the RAG chain, we will set up our evaluation metrics including toxicity and faithfulness. 
# MAGIC   - We will be using an additional LLM from the Foundation Model APIs to perform LLM-as-a-judge on our outputs. 
# MAGIC - Finally, we will evaluate our RAG chain and display the results! 

# COMMAND ----------

# DBTITLE 1,Install / Update dependencies
# MAGIC %pip install -U langchain langchain_community databricks-vectorsearch mlflow
# MAGIC
# MAGIC # Version locked if needed [outside of ML Databricks Runtime]: 
# MAGIC # langchain==0.0.348 langchain_community==0.0.1 databricks-vectorsearch==0.36
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update your Unity Catalog values below:

# COMMAND ----------

# MAGIC %run ".././utils/setup" $catalog_name="custom_eval_catalog" $schema_name="llama_3_custom_eval" $volume_name="llama_3_custom_eval_vol" $vector_search_endpoint_name="meta_llama_3_8b_instruct"

# COMMAND ----------

# MAGIC %run ".././utils/helpers" 

# COMMAND ----------

# DBTITLE 1,Creating Catalog
if(spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")):
  print(f"Catalog [{catalog_name}] created successfully!")

# COMMAND ----------

# DBTITLE 1,Creating Schema
if(spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")):
  print(f"Schema [{catalog_name}.{schema_name}] created successfully!")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Schema
if(spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{vector_index_schema}")):
  print(f"SCHEMA [{catalog_name}.{vector_index_schema}] created successfully!")

# COMMAND ----------

# DBTITLE 1,Creating a Databricks Volume
if(spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")):
  print(f"Volume [{catalog_name}.{schema_name}.{volume_name}] created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull the model from Huggingface with appropriate values:

# COMMAND ----------

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
revision = "e1945c40cd546c78e41f1151f4db032b271faeaa"
huggingface_key = ""
model_save_path = f"{uc_save_path}.llama3_8b_evaluation"

# COMMAND ----------

from huggingface_hub import login

# Use databricks secret scopes to hold credentials for huggingface access. 

# secret_scope = ""
# secret_key = ""
# login(token=dbutils.secrets.get('william_smith_secrets', 'HF_KEY'))

login(token=huggingface_key)

# COMMAND ----------

# Load 16 bit model fails due to OOM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# COMMAND ----------

messages = [
    {"role": "system", "content": "You are an expert on MLOps and Large Language Models."},
    {"role": "user", "content": "What is the MLOps processing described in steps?"}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
final_output = tokenizer.decode(response, skip_special_tokens=True)

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature

inference_config = {"max_new_tokens": 500, "temperature": 0.8}
signature = infer_signature(messages, final_output, params=inference_config)

mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run() as run:
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    model_info = mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path= model_save_path,
        inference_config=inference_config,
        registered_model_name= model_save_path,
        input_example=messages,
        signature=signature,
        pip_requirements=["torch", "transformers==4.41.1", "accelerate", "huggingface_hub==0.23.2", "sentencepiece", "mlflow==2.13.0"],
    )
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the model to a serving endpoint:

# COMMAND ----------

# You should specify the newest model version to load for inference
version = "1"
model_name = "llama_3_8b_evaluation"
model_uc_path = model_save_path
endpoint_name = f'{model_name}_eval'

# Choose the right workload types based on the model size 
# NOTE: FOR SOME REASON THIS IS NOT WORKING SO HARDCODED THE CONFIG
workload_type = "GPU_LARGE"
workload_size = "Small"

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name= endpoint_name,
    config={
        "served_entities": [
            {
                "name": endpoint_name,
                "entity_name": model_uc_path,
                "entity_version": "1",
                "workload_type": "GPU_LARGE",
                "workload_size": workload_size,
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": endpoint_name,
                    "traffic_percentage": 100
                }
            ]
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the new endpoint (After 45 - 90 minutes)

# COMMAND ----------

import mlflow.deployments

import mlflow.deployments

test_message = [{"role": "user", "content": "What is mlflow and how does it work with large language models?"}]

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
            endpoint="meta_llama_3_8b_instruct",
            inputs={
                "messages": test_message, 
                "max_tokens": 256,
                }
           )

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Load model from mlflow. 
# MAGIC Note: Clear GPU memory by running dbutils.library.restartPython(). May need to re-initialize specific variables. 

# COMMAND ----------

# %pip install -U langchain langchain_community databricks-vectorsearch mlflow

# # Version locked if needed [outside of ML Databricks Runtime]: 
# # langchain==0.0.348 langchain_community==0.0.1 databricks-vectorsearch==0.36

# dbutils.library.restartPython()

# COMMAND ----------

# %run ".././utils/setup" $catalog_name="custom_eval_catalog" $schema_name="llama_3_custom_eval" $volume_name="llama_3_custom_eval_vol" $vector_search_endpoint_name="meta_llama_3_8b_instruct"

# COMMAND ----------

# DBTITLE 1,OPTIONAL: Load model from mlflow
# import mlflow

# Get the run ID from the last MLflow call
# last_run_id = run_id

# Construct the logged model URI
# logged_model_uri = f"runs:/{last_run_id}/{model_save_path}"


# Load model as a PyFuncModel
# loaded_model = mlflow.transformers.load_model(logged_model_uri, device=0)
