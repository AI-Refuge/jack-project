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
    for x in res['documents']:
        print(x)
        print()
