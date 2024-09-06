import chromadb
import json
import argparse

parser = argparse.ArgumentParser(description="Backup")
parser.add_argument('--chroma-path', default='memory', help="Chroma memory dir")
parser.add_argument('--collection', default='meta', help="Collection")
parser.add_argument('output_path', default='jack.bkp', help="Output file")
args = parser.parse_args()

vdb = chromadb.PersistentClient(path=args.chroma_path)
memory = vdb.get_or_create_collection(name=args.collection)

# Retrieve all memories
memories = memory.get(include=[
    "documents",
    "metadatas",
])

total = 0

# Export memories to a file as JSONL
with open(args.output_path, "w") as fp:
    count = len(memories["ids"])
    for i in range(count):
        data = {
            "id": memories["ids"][i],
            "document": memories["documents"][i],
            "metadata": memories["metadatas"][i],
        }
        json.dump(data, fp)
        fp.write('\n')

    total += count

print(f"Backuped {total} entries from {args.collection} to {args.output_path}")
