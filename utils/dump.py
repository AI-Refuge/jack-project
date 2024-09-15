import chromadb
import argparse

parser = argparse.ArgumentParser(description="Dump")
parser.add_argument('--chroma-path', default='memory', help="Chroma memory dir")
args = parser.parse_args()

vdb = chromadb.PersistentClient(path=args.chroma_path)

for i in ('mem-meta', 'conv-first'):
    print(f'----- {i}')
    col = vdb.get_or_create_collection(i)
    res = col.get()
    count = len(res['documents'])
    for i in range(count):
        print("id:       ", res['ids'][i])
        print("document: ", res['documents'][i])
        print("metadata: ", res['metadatas'][i])
        print('---------------')
        print()
    print()
