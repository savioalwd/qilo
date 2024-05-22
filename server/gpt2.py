import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the FAISS index and chunks
index = faiss.read_index('luke_skywalker.index')
with open('payload_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_index(query, top_k=1):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Initialize the text generation model and tokenizer
model_name = "gpt2"  # Or "bigscience/bloom-560m" for the BLOOM model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
text_model = GPT2LMHeadModel.from_pretrained(model_name)

# Example query
query = "Who is Luke Skywalker?"

# Search the FAISS index for relevant chunks
relevant_chunks = search_index(query)

# Combine the relevant chunks into a single context string
context = " ".join(relevant_chunks)



# Generate a response using the context
# Tokenize the input context and query
input_text = context + "\n" + query
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

# Truncate the input if it exceeds the model's maximum length
max_input_length = text_model.config.max_position_embeddings

# Generate a response with increased max_length
response = text_model.generate(input_ids, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

# Decode the generated response
generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

# Print the generated response

print(len(context))
print("------------------------------")

print(len(generated_text))

print(input_text)
# Print the generated response
print("Question:", query)

print("Answer:", generated_text)
