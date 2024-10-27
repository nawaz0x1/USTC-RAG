from utils.embedding import Embedding
from dotenv import load_dotenv

def perform_embedding(data_path = 'data', chunk_size = 500, chunk_overlap = 50, index_name = None, model = "nvidia/llama-3.2-nv-embedqa-1b-v1"):
    load_dotenv()
    
    try:
        embedding = Embedding(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        embedding.load_data(data_path)
        embedding.store_vectors(index_name = index_name, model = model)
    except Exception as e:
        print(e)
        
    
    
if __name__ == '__main__':
    perform_embedding(data_path='../data', index_name="ustc-rag-2048")