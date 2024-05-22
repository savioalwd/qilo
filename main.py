import faiss
import torch
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# Initialize FastAPI
app = FastAPI()

# Enable CORS
origins = ["*"]  # Adjust as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the FAISS index and chunks
index = faiss.read_index('luke_skywalker.index')
with open('payload_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Load the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the BART model for question answering
bloom_model_name = "facebook/bart-large"
bloom_tokenizer = AutoTokenizer.from_pretrained(bloom_model_name)
bloom_model = AutoModelForQuestionAnswering.from_pretrained(bloom_model_name)

# Initialize the GPT-2 model and tokenizer for text generation
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_text_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

def search_index(query, top_k=1):
    """Search the FAISS index for relevant chunks."""
    query_vector = sentence_model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

class GenerateRequest(BaseModel):
    """Pydantic model for the request body."""
    question: str
    model: str
    max_new_tokens: int

@app.post("/generate")
def generate_response(request: GenerateRequest):
    """Generate response based on the provided model."""
    question = request.question
    model = request.model
    max_new_tokens = request.max_new_tokens

    # Check if the model is supported
    if model.lower() not in ["bloom", "gpt2", "openai"]:
        raise HTTPException(status_code=400, detail="Unsupported model")

    # Search the FAISS index for relevant chunks
    relevant_chunks = search_index(question)

    # Combine the relevant chunks into a single context string
    context = " ".join(relevant_chunks)

    if model.lower() == "bloom":
        # Generate response using the BART model for question answering
        input_text = question + " " + context
        inputs = bloom_tokenizer.encode_plus(input_text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = bloom_model(input_ids=input_ids, attention_mask=attention_mask)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = bloom_tokenizer.convert_tokens_to_string(bloom_tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    elif model.lower() == "gpt2":
        # Generate response using the GPT-2 model for text generation
        input_text = context + "\n" + question
        input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        response = gpt2_text_model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=gpt2_tokenizer.eos_token_id, num_return_sequences=1)
        answer = gpt2_tokenizer.decode(response[0], skip_special_tokens=True)
    else:
        # Generate response using OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=context + "\n" + question,
            max_tokens=max_new_tokens
        )
        answer = response.choices[0].text.strip()

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
