import chromadb
import argparse
import json
from datetime import datetime, timezone

ts = int(datetime.now(timezone.utc).timestamp())

parser = argparse.ArgumentParser(description="Load")
parser.add_argument('--chroma-path', default='memory', help="Chroma memory dir")
parser.add_argument('--collection', default='mem-meta', help="Collection")
parser.add_argument('--name', default='@jack', help="Meta name of memory")
parser.add_argument('input_path', default='jack.bkp', help="Memory JSONL file")
args = parser.parse_args()

vdb = chromadb.PersistentClient(path=args.chroma_path)
memory = vdb.get_or_create_collection(
    name=args.collection,
    metadata={
        "timestamp": ts,
        "name": args.name,
    },
)

if memory.count() != 0:
    # meta: Overriding is not recommended
    #   unless you know what you are doing
    print(f"Memory collection {args.collection} not empty")
    exit()

total = 0

# Import memories from JSONL
with open(args.input_path, "r") as f:
    data = [json.loads(line) for line in f]
    memory.add(
        ids=[d['id'] for d in data],
        metadatas=[d['metadata'] for d in data],
        documents=[d['document'] for d in data],
    )
    total = len(data)

print(f"Loaded {total} into {args.collection} entries from {args.input_path}")
