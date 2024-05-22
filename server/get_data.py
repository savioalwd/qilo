import requests
from bs4 import BeautifulSoup
import re

url = "https://en.wikipedia.org/wiki/Luke_Skywalker"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

content = soup.find_all("p")

#getting only the text content
text = "\n".join([p.get_text() for p in content])

#cleaning the text
pattern = r'\[\d+\]'
payload = re.sub(pattern, '', text)

print(payload)
#function for Chunk the payload
def chunk_text(text, chunk_size=512):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in text.split('. '):
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

#getting the chunks from the string as dictionary by calling the  chunk_text
chunks = chunk_text(payload)

#importing the faiss and transformers library

import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2') #specific pre-trained model that is lightweight and fast, suitable for generating sentence embeddings.

#Generating Embeddings for Text Chunks
embeddings = model.encode(chunks)



#Populating faiss Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index and chunks for later use
faiss.write_index(index, 'luke_skywalker.index')

import pickle
with open('payload_chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

