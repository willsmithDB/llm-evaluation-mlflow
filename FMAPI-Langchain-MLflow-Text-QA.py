# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation of RAG (Retrieval-Augmented Generation) chain using Databricks Foundation Model APIs and MLflow!
# MAGIC
# MAGIC - We will use langchain to pull MLflow documentation and chunk it. 
# MAGIC - We will use the Databricks Foundation Model APIs to automatically compute embeddings from the chunks. 
# MAGIC - We will then create an index within a Databricks Vector Search index to hold the embeddings and act as a retriever for our RAG chain. 
# MAGIC - DBRX from the Databricks Foundation Model APIs will be our primary model for our RAG chain.
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

# MAGIC %run "./utils/setup" $catalog_name="CATALOG" $schema_name="fmapi_eval" $volume_name="fmapi_vol" $vector_search_endpoint_name="VECTOR_SEARCH"

# COMMAND ----------

# MAGIC %run "./utils/helpers" 

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

from pyspark.sql.functions import col, monotonically_increasing_id
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://mlflow.org/docs/latest/index.html")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=32)
docs = text_splitter.split_documents(documents)


df = spark.createDataFrame(docs).drop(col("metadata")).withColumn("id", monotonically_increasing_id())

display(df)

df.write.option("mergeSchema", "true").mode("overwrite").format("delta").saveAsTable(f"{uc_save_path}.raw_mlflow_docs")

# COMMAND ----------


alter_table_cdf = f"ALTER TABLE {uc_save_path}.raw_mlflow_docs SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
 
if(spark.sql(alter_table_cdf)):
  print(f"Table [{uc_save_path}.raw_mlflow_docs] updated successfully!")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

create_vector_search_if_not_exists(vector_search_endpoint_name, vsc)

# COMMAND ----------

create_vector_index_if_not_exists(vector_search_endpoint_name, vector_index_name, uc_save_path, embeddings_model, vsc)

# COMMAND ----------

vs_index = vsc.get_index(endpoint_name= vector_search_endpoint_name, index_name= vector_index_name)

vs_index.describe()

# COMMAND ----------

results = vs_index.similarity_search(
    query_text="Gen AI",
    columns=["id"
             , "page_content"],
    num_results=2
    )

display(results)

# COMMAND ----------

from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


def get_retriever(persist_dir: str = None):
    
    vs_index = vsc.get_index(
        endpoint_name= vector_search_endpoint_name,
        index_name= vector_index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="page_content"
    )
    return vectorstore.as_retriever()


# test our retriever
retriever = get_retriever()

# COMMAND ----------

# If running a Databricks notebook attached to an interactive cluster in "single user"
# or "no isolation shared" mode, you only need to specify the endpoint name to create
# a `Databricks` instance to query a serving endpoint in the same workspace.

# Otherwise, you can manually specify the Databricks workspace hostname and personal access token
# or set `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables, respectively.
# You can set those environment variables based on the notebook context if run on Databricks

import os
from langchain_community.llms import Databricks

# Need this for job run: 
# os.environ['DATABRICKS_URL'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
# os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

from langchain.llms import Databricks
from langchain_core.messages import HumanMessage, SystemMessage

def transform_input(**request):
  request["messages"] = [
    {
      "role": "user",
      "content": request["prompt"]
    }
  ]
  del request["prompt"]
  return request

# databricks-meta-llama-3-70b-instruct or databricks-dbrx-instruct
# llm = Databricks(endpoint_name="databricks-dbrx-instruct", transform_input_fn=transform_input, extra_params={"temperature": 0.1, "max_tokens":1000})
llm = Databricks(endpoint_name=llm_model, transform_input_fn=transform_input, extra_params={"temperature": 0.1, "max_tokens":512})

#if you want answers to generate faster, set the number of tokens above to a smaller number
prompt = "What is Generative AI?"

displayHTML(llm(prompt))

# COMMAND ----------

from langchain import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain():
  
  template = """You are a life sciences researcher with deep expertise in cystic fibrosis and related comorbidities. Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  Use only information in the following paragraphs to answer the question. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.

  {context}
  
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever= retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt
    }
  )
  
  # Set verbose=True to see the full prompt:
  return qa_chain

# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

question = "How would one evaluate gen ai models with mlflow?"

result = qa_chain({"query": question})
# Check the result of the query
print(result["result"])
# Check the source document from where we draw from 

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

# COMMAND ----------

import pandas as pd

eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "What is Spark?",
            "How to run mlflow.evaluate()?",
            "How to log_table()?",
            "How to load_table()?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
            "To use mlflow.evaluate(), you typically need to follow these steps:?"
            " Initialize MLflow: First, you need to initialize MLflow by starting a run. This can be done using mlflow.start_run()."
            " Define Your Model and Metrics: Train your machine learning model and evaluate it on your test dataset. Compute the metrics you're interested in."
            " Log Metrics: Log your evaluation metrics using mlflow.log_metric()."
            " Finish Run: Once you've logged all the necessary metrics, you can end the MLflow run with mlflow.end_run().",
            "To log a table with MLflow using mlflow.log_table(), you need to first construct your table data in a format that can be logged, such as a Pandas DataFrame or a list of dictionaries. Then, you can use the mlflow.log_table() function to log this data."
            "Here's a general outline of how to use mlflow.log_table():"
            "Prepare your table data: You need to have your table data ready in a suitable format, such as a Pandas DataFrame or a list of dictionaries."
            "Log the table: Use mlflow.log_table() to log the table data. Provide a name for the table and the data itself.",
            "To load a table logged with MLflow using mlflow.load_table(), you need to provide the run ID and the name of the table you want to load. This function allows you to retrieve the logged table data as a Pandas DataFrame."
        ],
    }
)

# COMMAND ----------

def model(input_df):
    answer = []
    for index, row in input_df.iterrows():
        answer.append(qa_chain(row["questions"]))

    return answer

# COMMAND ----------

question = "What is mlflow.evaluate() ?"
result = qa_chain({"query": question})
# Check the result of the query
print(result["result"])
# Check the source document from where we 

# COMMAND ----------

from mlflow.metrics.genai import faithfulness, EvaluationExample

# Create a good and bad example for faithfulness in the context of this problem
faithfulness_examples = [
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. ",
        score=2,
        justification="The output provides a working solution, using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions.",
        score=5,
        justification="The output provides a solution that is using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
]

faithfulness_metric = faithfulness(
    model="endpoints:/databricks-llama-2-70b-chat", examples=faithfulness_examples
)
print(faithfulness_metric)

# COMMAND ----------

from mlflow.metrics.genai import relevance, EvaluationExample

relevance_metric = relevance(model="endpoints:/databricks-llama-2-70b-instruct")

print(relevance_metric)

# COMMAND ----------

import mlflow 

results = mlflow.evaluate(
    model,
    eval_df,
    targets="ground_truth",
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "source_documents",
        }
    },
)
print(results.metrics)

# COMMAND ----------

print(results.metrics)

# COMMAND ----------

results.tables["eval_results_table"]
