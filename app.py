from tabnanny import verbose
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from llama_index import VectorStoreIndex
from llama_index.vector_stores import SingleStoreVectorStore
from llama_index.llms import ChatMessage, MessageRole
import gradio as gr
import os

os.environ["OPENAI_API_KEY"] = "sk-xxx"  # Replace with your key
os.environ["SINGLESTOREDB_URL"] = "admin:password@svc-e6090846-4556-4e7f-b98b-28ae9c6aec5f-dml.aws-oregon-3.svc.singlestore.com:3306"
vector_store = SingleStoreVectorStore(table_name="embeddings5", database="demo")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

chat_engine = index.as_chat_engine(
        chat_mode='context',
        verbose=True,
)

def predict(message, history):
    history_llamaindex_format = []
    for human, ai in history:
        history_llamaindex_format.append(ChatMessage(content=human, role=MessageRole.USER))
        history_llamaindex_format.append(ChatMessage(content=ai, role=MessageRole.ASSISTANT))
    history_llamaindex_format.append(ChatMessage(content=message, role=MessageRole.USER))

    return chat_engine.chat(message=message, chat_history=history_llamaindex_format).response
    

demo = gr.ChatInterface(predict)

if __name__ == "__main__":
    demo.queue().launch()