import ollama
import os
import json
import constant

def getEmbeddingsFromPrompts(filename, chunks):
    print("try to get embeddings...")
    if (embeddings := loadEmbeddings(filename)) is not False:
        print("... loaded!")
        return embeddings
    print("asking to ollama...")
    embeddings = []
    for chunk in chunks:
        embedding = getEmbeddingFromPrompt(chunk)
        embeddings.append(embedding)
    print("saving...")
    saveEmbeddings(filename, embeddings)
    return embeddings

def getEmbeddingFromPrompt(prompt):
    prompt = str(prompt)
    return ollama.embeddings(model=constant.model, prompt=prompt)[
        "embedding"        
    ]
    
def saveEmbeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    filePath = "embeddings/" + filename + ".json"
    with open(filePath, "w") as file:
        json.dump(embeddings, file)

def loadEmbeddings(filename):
    filepath = "embeddings/" + filename + ".json"
    if not os.path.exists(filepath):
        return False
    with open(filepath, "r") as f:
        return json.load(f)