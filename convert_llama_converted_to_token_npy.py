import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,LlamaModel

pretrained_dir = 'llama_converted'
device = 'cuda'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained(pretrained_dir).to(device) 
print(len(model.layers))

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Extract token embeddings
embeddings = model.embed_tokens.weight.cpu().detach().numpy()

# Save the embeddings to a numpy file
np.save('llama-token.npy', embeddings)
print('Embeddings saved')