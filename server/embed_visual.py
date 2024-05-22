import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example list of text chunks
chunks = [
    "Luke Skywalker is a fictional character and the main protagonist of the original film trilogy of the Star Wars franchise created by George Lucas.",
    "He was played by Mark Hamill in the original trilogy and in the sequel trilogy.",
    "Luke is a Jedi Master and is the twin brother of Princess Leia Organa."
]

# Generate embeddings
embeddings = model.encode(chunks)

# Reduce dimensionality using t-SNE with a lower perplexity
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the 2D embeddings
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='blue')

# Annotate the points with text chunk index
for i, txt in enumerate(range(embeddings_2d.shape[0])):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("2D Visualization of Text Chunk Embeddings using t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()
