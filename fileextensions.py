import os

def getAllFilesFromDirectory(dir):
    return os.listdir(dir)

def parseFile(dir, filename):
    with open(dir + filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            #pline = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
    return paragraphs