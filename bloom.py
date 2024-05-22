import faiss
import torch
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the FAISS index and chunks
index = faiss.read_index('luke_skywalker.index')
with open('payload_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Bloom model
bloom_model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(bloom_model_name)
bloom_model = AutoModelForQuestionAnswering.from_pretrained(bloom_model_name)

def search_index(query, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

def generate_response(question, context):
    input_text = question + " " + context
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=1024, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = bloom_model(input_ids=input_ids, attention_mask=attention_mask)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    return answer

# Example question
question = "Who is Luke Skywalker?"

# Search the FAISS index for relevant chunks
relevant_chunks = search_index(question)

# Combine the relevant chunks into a single context string
context = " ".join(relevant_chunks)

# Split the context into smaller chunks to avoid truncation
max_chunk_length = 512  # Adjust as needed
context_chunks = [context[i:i+max_chunk_length] for i in range(0, len(context), max_chunk_length)]

# Generate responses for each chunk
responses = []
for chunk in context_chunks:
    response = generate_response(question, chunk)
    responses.append(response)

# Combine responses from different chunks
response = " ".join(responses)

print(context)
# Print the generated response
print("Question:", question)

print("Answer:", response)
