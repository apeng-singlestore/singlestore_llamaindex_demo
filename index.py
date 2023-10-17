from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import SingleStoreVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SimpleNodeParser
import os
from dotenv import load_dotenv
load_dotenv()

# load documents
print("Parsing documents...")
data_dir = "./data/"
documents = SimpleDirectoryReader(data_dir).load_data()
parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)

nodes = parser.get_nodes_from_documents(documents)

print("Ingesting documents...")
# Create an index over the documents
vector_store = SingleStoreVectorStore(table_name="embeddings", database="demo")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("Index created.")

