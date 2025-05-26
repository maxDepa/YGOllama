import numpy as np
from numpy import linalg
import fileextensions as fe
import constant
import embedding
import ollama
from integration import message as msg

SYSTEM_PROMPT = constant.systemPrompt

def main():
    print("loading data set...")
    # paragraphs = getCardParagraphs()
    # embeddings = embedding.getEmbeddingsFromChunks("cards", paragraphs)
    
    paragraphs = getMechanicsParagraphs()
    embeddings = embedding.getEmbeddingsFromChunks("mechanics", paragraphs)
    
    while(True):
        prompt = input("How can I help you? ")
        promptEmbedding = embedding.getEmbeddingFromPrompt(prompt)
        most_similar_chunks = findMostSimilarChunk(promptEmbedding, embeddings)[:10]
        
        print("create context")
        context = "\n"
        for item in most_similar_chunks:
            similarityScore = item[0]
            index = item[1]
            context.join(paragraphs[index])
            print("Score: " + str(similarityScore) + "\n" + str(paragraphs[index]))
        
        print("asking response")
        response = ollama.chat(
            model=constant.model,
            temperature=0,
            messages=[ 
                    msg.create("system",SYSTEM_PROMPT + context),
                    msg.create("user", prompt)
            ]
        )
        # response = ollama.generate(
        #     #model="llama2",
        #     model=constant.model,
        #     prompt=f"Using this data: {context}. Respond to this prompt: {prompt}",
        #     temperature=0
        #     )

        #print(response["message"]["content"])
        print(response['response'])

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

def findMostSimilarChunk(needle, haystack):
    needle_norm = linalg.norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * linalg.norm(item)) for item in haystack        
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

if __name__ == "__main__":
    main()