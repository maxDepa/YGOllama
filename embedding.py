import ollama
import os
import json
import constant

def getEmbeddingsFromChunks(filename, chunks):
    #try to load
    if (embeddings := loadEmbeddings(filename)) is not False:
        return embeddings

    # getting them from ollama
    embeddings = []
    for chunk in chunks:
        embedding = getEmbeddingFromPrompt(chunk)
        embeddings.append(embedding)
    
    # saving
    saveEmbeddings(filename, embeddings)
    return embeddings

def getEmbeddingFromPrompt(prompt):
    prompt = str(prompt)
    return ollama.embeddings(model=constant.model, prompt=prompt)[
        "embedding"        
    ]
    
def saveEmbeddings(filename, embeddings):
    if not os.path.exists(constant.embeddingsDirectory):
        os.makedirs(constant.embeddingsDirectory)
    filepath = getEmbeddingJsonPath(filename)
    with open(filepath, "w") as file:
        json.dump(embeddings, file)

def loadEmbeddings(filename):
    filepath = getEmbeddingJsonPath(filename)
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r") as f:
        return json.load(f)
    
def getEmbeddingJsonPath(filename):
    return constant.embeddingsDirectory + filename + ".json"