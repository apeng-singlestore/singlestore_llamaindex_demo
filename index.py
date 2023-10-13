from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import SingleStoreVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SimpleNodeParser
import os
import openai

openai.api_key = "sk-xxx"

# load documents
data_dir = "./data/"
documents = SimpleDirectoryReader(data_dir).load_data()
parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)

nodes = parser.get_nodes_from_documents(documents)
print("Document ID:", documents[0].doc_id)

# Create an index over the documents
os.environ["SINGLESTOREDB_URL"] = "admin:password@svc-e6090846-4556-4e7f-b98b-28ae9c6aec5f-dml.aws-oregon-3.svc.singlestore.com:3306"
vector_store = SingleStoreVectorStore(table_name="embeddings5", database="demo")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("Index created")

