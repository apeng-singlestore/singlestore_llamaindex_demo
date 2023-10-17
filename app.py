from tabnanny import verbose
from llama_index import VectorStoreIndex
from llama_index.vector_stores import SingleStoreVectorStore
from llama_index.llms import ChatMessage, MessageRole
import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

vector_store = SingleStoreVectorStore(table_name="embeddings", database="demo")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

chat_engine = index.as_chat_engine(
        chat_mode='openai',
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