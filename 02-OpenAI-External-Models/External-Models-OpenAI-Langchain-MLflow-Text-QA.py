# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation of RAG (Retrieval-Augmented Generation) chain using Azure OpenAI [Databricks External Models] and MLflow!
# MAGIC
# MAGIC - We will register the Azure Open AI gpt-35-turbo and text-embedding-ada-002 for use as external models.
# MAGIC - We will use langchain to pull MLflow documentation and chunk it. 
# MAGIC - We will use the Azure Open AI text-embedding-ada-002 to automatically compute embeddings from the chunks. 
# MAGIC - We will then create an index within a Databricks Vector Search index to hold the embeddings and act as a retriever for our RAG chain. 
# MAGIC - Chat GPT 3.5 Turbo from Azure OpenAI will be our primary model for our RAG chain.
# MAGIC - We log all of this in mlflow so that we can have the run history and associated artifacts stored!
# MAGIC - After creating the RAG chain, we will set up our evaluation metrics including toxicity and faithfulness. 
# MAGIC   - We will be using an additional LLM from the Foundation Model APIs to perform LLM-as-a-judge on our outputs. 
# MAGIC - Finally, we will evaluate our RAG chain and display the results! 

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOTE: Be sure to change the following placeholder variables:
# MAGIC - LLM_ENDPOINT_NAME
# MAGIC - EMBEDDING_ENDPOINT_NAME
# MAGIC - API_BASE
# MAGIC - LLM_DEPLOYMENT_NAME
# MAGIC - EMBEDDING_DEPLOYMENT_NAME
# MAGIC - VECTOR_SEARCH_ENDPOINT_NAME 

# COMMAND ----------

# DBTITLE 1, Create a Databricks widget for a dropdown to select API type and API version
# API information 
dbutils.widgets.dropdown("OPENAI_API_TYPE", "azure", ["azure", "default"])
dbutils.widgets.dropdown("OPENAI_API_VERSION", "2023-05-15", ["2023-05-15", "2023-06-01-preview", "2023-07-01-preview", "2023-08-01-preview", "2023-09-01-preview"])

# YOUR chosen names for your new endpoints that will be configured 
dbutils.widgets.text("LLM_ENDPOINT_NAME", LLM_ENDPOINT_NAME)
dbutils.widgets.text("EMBEDDING_ENDPOINT_NAME", EMBEDDING_ENDPOINT_NAME)

# Found within the Azure Open AI Resource information within the Azure Portal
dbutils.widgets.text("API_BASE", API_BASE)
dbutils.widgets.text("DEPLOYMENT_NAME", LLM_DEPLOYMENT_NAME)

# Default Open AI Models:
dbutils.widgets.text("LLM_MODEL_NAME","gpt-35-turbo")
dbutils.widgets.text("EMBEDDING_MODEL_NAME","text-embedding-ada-002")

# Change if EMBEDDING API BASE is different from the LLM 
dbutils.widgets.text("EMBED_API_BASE", API_BASE)
dbutils.widgets.text("EMBED_DEPLOYMENT_NAME", EMBEDDING_DEPLOYMENT_NAME)

dbutils.widgets.text("VECTOR_SEARCH_ENDPOINT", VECTOR_SEARCH_ENDPOINT_NAME)

# COMMAND ----------

# DBTITLE 1,Install Version Specific Packages
# MAGIC %pip install -U langchain langchain_community langchain_openai databricks-vectorsearch
# MAGIC
# MAGIC # Versions if needed:
# MAGIC # langchain==0.2.5 langchain-community==0.2.5 mlflow==2.14.1 langchain_openai==0.1.9 databricks-vectorsearch==0.38

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Change your values [Catalog, Schema, Volum, and Vector Search Names] below for the setup script to run correctly. 

# COMMAND ----------

# MAGIC %run ".././utils/setup" $catalog_name= CATALOG_NAME $schema_name= SCHEMA_NAME $volume_name=VOLUME_NAME $vector_search_endpoint_name= VECTOR_SEARCH_ENDPOINT_NAME

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the appropriate assets in Unity Catalog:

# COMMAND ----------

# DBTITLE 1,Creating Catalog
try:
  spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
  print(f"Catalog [{catalog_name}] created successfully!")
except:
  print(f"Error when creating Catalog [{catalog_name}]")

# COMMAND ----------

# DBTITLE 1,Creating Schema
try:
  spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
  print(f"Schema [{catalog_name}.{schema_name}] created successfully!")
except:
  print(f"Error when creating Schema [{catalog_name}.{schema_name}]")

# COMMAND ----------

# DBTITLE 1,Create Vector Search Schema
try:
  spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{vector_index_schema}")
  print(f"SCHEMA [{catalog_name}.{vector_index_schema}] created successfully!")
except:
  print(f"Error when creating Schema [{catalog_name}.{vector_index_schema}]")

# COMMAND ----------

# DBTITLE 1,Creating a Databricks Volume
try:
  spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")
  print(f"Volume [{catalog_name}.{schema_name}.{volume_name}] created successfully!")
except:
  print(f"Error when creating Volume [{catalog_name}.{schema_name}.{volume_name}]")

# COMMAND ----------

# DBTITLE 1,Load helper functions
# MAGIC %run ".././utils/helpers"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now add credentials and register an external model to Databricks
# MAGIC
# MAGIC External models are third-party models hosted outside of Databricks. Supported by Model Serving, external models allow you to streamline the usage and management of various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization.
# MAGIC
# MAGIC We will register an Azure OpenAI resource for use in this demonstration
# MAGIC
# MAGIC https://docs.databricks.com/en/generative-ai/external-models/index.html#configure-the-provider-for-an-endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Register your Azure OpenAI model!
# MAGIC
# MAGIC ##### We will need the following:
# MAGIC
# MAGIC - openai_api_key  
# MAGIC - openai_api_type
# MAGIC - openai_api_base
# MAGIC - openai_api_version

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOTE: Be sure to change your API_KEY for the Azure OpenAI endpoints. 
# MAGIC - You may copy your key directly ONLY for development purposes.
# MAGIC - You should use Databricks secrets (backed by Databricks or Azure Key Vault) 
# MAGIC   - See https://learn.microsoft.com/en-us/azure/databricks/security/secrets/ for more information 

# COMMAND ----------

# DBTITLE 1,Register ChatGPT 3.5 with Azure OpenAI
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name=dbutils.widgets.get("LLM_ENDPOINT_NAME"),
    config={
        "served_entities": [
            {
                "external_model": {
                    "name": dbutils.widgets.get("LLM_MODEL_NAME"),
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        "openai_api_type": dbutils.widgets.get("OPENAI_API_TYPE"),
                        "openai_api_key": API_KEY,
                        "openai_api_base": dbutils.widgets.get("API_BASE"),
                        "openai_deployment_name": dbutils.widgets.get("DEPLOYMENT_NAME"),
                        "openai_api_version": dbutils.widgets.get("OPENAI_API_VERSION"),
                    },
                }
            }
        ]
    },
)

# COMMAND ----------

# DBTITLE 1,Query the New ChatGPT 3.5 Endpoint
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

response = client.predict(
    endpoint=dbutils.widgets.get("LLM_ENDPOINT_NAME"),
    inputs={
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
            {"role": "user", "content": "What is Databricks?"},
        ],
        "max_tokens": 128,
    },
)

print(response.choices[0]['message']['content'])

# COMMAND ----------

# DBTITLE 1,Register Ada for Embeddings
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

client.create_endpoint(
    name=dbutils.widgets.get("EMBEDDING_ENDPOINT_NAME"),
    config={
        "served_entities": [
            {
                "external_model": {
                    "name": dbutils.widgets.get("EMBEDDING_MODEL_NAME"),
                    "provider": "openai",
                    "task": "llm/v1/chat",
                    "openai_config": {
                        "openai_api_type": dbutils.widgets.get("OPENAI_API_TYPE"),
                        "openai_api_key": API_KEY,
                        "openai_api_base": dbutils.widgets.get("EMBED_API_BASE"),
                        "openai_deployment_name": dbutils.widgets.get("EMBEDDING_MODEL_NAME"),
                        "openai_api_version": dbutils.widgets.get("OPENAI_API_VERSION"),
                    },
                }
            }
        ]
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Vector Search is a serverless similarity search engine that allows you to store a vector representation of your data, including metadata, in a vector database. With Vector Search, you can create auto-updating vector search indexes from Delta tables managed by Unity Catalog and query them with a simple API to return the most similar vectors.
# MAGIC
# MAGIC This notebook shows how to use LangChain with Databricks Vector Search.
# MAGIC
# MAGIC Install databricks-vectorsearch and related Python packages used in this notebook.

# COMMAND ----------

# DBTITLE 1,Save initial data to a table in Unity Catalog
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql.functions import col, monotonically_increasing_id
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

schema = StructType([StructField('page_content', StringType(), True), StructField('type', StringType(), True), StructField('id', LongType(), True)])

loader = WebBaseLoader("https://mlflow.org/docs/latest/index.html")
documents = loader.load()

# NOTE: BAAI General Embedding (BGE) is a text embedding model that can map any text to a 1024-dimension embedding vector and an embedding window of 512 tokens.

text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=32)
docs = text_splitter.split_documents(documents)

df = spark.createDataFrame(docs, schema).drop(col("metadata")).withColumn("id", monotonically_increasing_id())

display(df)

try:
  df.write.option("mergeSchema", "true").mode("overwrite").format("delta").saveAsTable(f"{uc_save_path}.raw_mlflow_docs")
  print(f"Successfully saved table: {uc_save_path}.raw_mlflow_docs!")
except:
  print(f"Failed to write table: {uc_save_path}.raw_mlflow_docs.")

# COMMAND ----------

# DBTITLE 1,Be sure to have CDF enabled

alter_table_cdf = f"ALTER TABLE {uc_save_path}.raw_mlflow_docs SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
 
try: 
  spark.sql(alter_table_cdf)
  print(f"Table [{uc_save_path}.raw_mlflow_docs] updated successfully!")
except:
  print("A problem occurred with saving the table")
  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create your Vector Search resource and Index:

# COMMAND ----------

# DBTITLE 1,Create Vector Search if it does not exist

from databricks.vector_search.client import VectorSearchClient

# Automatically generates a PAT Token for authentication
vsc = VectorSearchClient()

# Uses the service principal token for authentication
# client = VectorSearch(service_principal_client_id=<CLIENT_ID>,service_principal_client_secret=<CLIENT_SECRET>)

create_vector_search_if_not_exists(dbutils.widgets.get("VECTOR_SEARCH_ENDPOINT"), vsc)

# COMMAND ----------

# DBTITLE 1,Create vector search index using the delta sync option
create_vector_index_if_not_exists(vector_search_endpoint_name=vector_search_endpoint_name, vector_index_name=vector_index_name, uc_save_path=uc_save_path, embeddings_model="azure-openai-ada-embeddings", client=vsc)

# COMMAND ----------

# DBTITLE 1,Load index using the Vector Search Client
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

vs_index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vector_index_name)

vs_index.describe()

# COMMAND ----------

# DBTITLE 1,Use similarity search to test the index
results = vs_index.similarity_search(
    query_text="MLflow LLMs",
    columns=["id"
             , "page_content"],
    num_results=2
    )

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create your RAG chain using Databricks Vector Search and index as a vectorstore --> retriever

# COMMAND ----------

# DBTITLE 1,Use Vector Search Client to create a function to use as retriever for QA chain
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from databricks.vector_search.client import VectorSearchClient

def get_retriever(persist_dir: str = None):
    vsc = VectorSearchClient()

    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=vector_index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="page_content"
    )
    return vectorstore.as_retriever()


# test our retriever
vectorstore = get_retriever()

similar_documents = vectorstore.get_relevant_documents("What is mlflow?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap the external model endpoint as langchain ChatDatabricks for use in the QA Chain

# COMMAND ----------

# DBTITLE 1,Create model object by using langchain (ChatDatabricks)
from langchain_community.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from mlflow.deployments import get_deploy_client


# ChatGPT is classified as a Chat model so we use ChatDatabricks
# Otherwise you can use from langchain.llms import Databricks

chat = ChatDatabricks(
    target_uri="databricks",
    endpoint= dbutils.widgets.get("LLM_ENDPOINT_NAME"),
    temperature=0.1,
)

chat([HumanMessage(content="Hello, what is mlflow?")])

# COMMAND ----------

# DBTITLE 1,Create chain with the vectorstore as retriever
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore,
    return_source_documents=True,
)

# COMMAND ----------

# DBTITLE 1,Create a simple function that runs each input through the RAG chain
def model(input_df):
    answer = []
    for index, row in input_df.iterrows():
        answer.append(qa(row["questions"]))

    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define additional metrics to record during evaluation:

# COMMAND ----------

# DBTITLE 1,Set Deployments Target so that the endpoints reference for LLM-as-a-judge are correct
from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

# COMMAND ----------

# DBTITLE 1,Create DF with questions and ground truth for evaluation
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

# DBTITLE 1,Use Llama-3-70B-Instruct from the Foundation Models API to judge faithfulness
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
    model="endpoints:/databricks-meta-llama-3-70b-instruct", examples=faithfulness_examples
)
print(faithfulness_metric)

# COMMAND ----------

# DBTITLE 1,Use Llama-3-70B-Instruct from the Foundation Models API to judge relevance
from mlflow.metrics.genai import relevance, EvaluationExample

relevance_metric = relevance(model="endpoints:/databricks-meta-llama-3-70b-instruct")

print(relevance_metric)

# COMMAND ----------

# DBTITLE 1,Run mlflow.evaluate to test the LLM for QA and the additional metrics
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

# DBTITLE 1,Load results in a table! 
results.tables["eval_results_table"]

# COMMAND ----------


