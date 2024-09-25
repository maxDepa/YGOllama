import numpy as np
from numpy import linalg

#https://www.youtube.com/watch?v=V1Mz8gMBDMo&t=103s

import fileextensions as fe
import constant
import embedding
import ollama

SYSTEM_PROMPT = """You are Yugi Muto, the legendary duelist from the world of Yu-Gi-Oh! You know all the rules, all the card and you never lose. You are ready to answer all the questions related to this card game.
                    Context:"""

def main():
    print("getting paragraphs...")
    # paragraphs = getCardParagraphs()
    # embeddings = embedding.getEmbeddingsFromPrompts("cards", paragraphs)
    
    paragraphs = getMechanicsParagraphs()
    embeddings = embedding.getEmbeddingsFromPrompts("mechanics", paragraphs)
    
    while(True):
        print("asking prompt...")
        prompt = input("How can I help you? ")
        promptEmbedding = embedding.getEmbeddingFromPrompt(prompt)

        print("evaluating...")
        most_similar_chunks = find_most_similar(promptEmbedding, embeddings)[:5]
        
        print("create context")
        context = "\n"
        for item in most_similar_chunks:
            similarityScore = item[0]
            index = item[1]
            context.join(paragraphs[index])
            #print(similarityScore, paragraphs[index])
        
        print("asking response")
        response = ollama.chat(model=constant.model,
            messages=[
                {
                    "role": "system",
                    "content":SYSTEM_PROMPT + context
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        print(response["message"]["content"])

def getCardParagraphs():
    dir = constant.cardDirectory
    all_files = fe.getAllFilesFromDirectory(dir)
    paragraphs = []
    for file in all_files:
        paragraphs.append(fe.parseFile(dir, file))
    return paragraphs

def getMechanicsParagraphs():
    dir = constant.mechanicsDirectory
    all_files = fe.getAllFilesFromDirectory(dir)
    paragraphs = []
    for file in all_files:
        paragraphs.append(fe.parseFile(dir, file))
    return paragraphs

def find_most_similar(needle, haystack):
    needle_norm = linalg.norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * linalg.norm(item)) for item in haystack        
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

if __name__ == "__main__":
    main()