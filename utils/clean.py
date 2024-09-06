import chromadb
from datetime import datetime, timezone
import argparse

ts = int(datetime.now(timezone.utc).timestamp())

parser = argparse.ArgumentParser(description="Clean")
parser.add_argument('--chroma-path', default='memory', help="Chroma memory dir")
parser.add_argument('--collection', default='meta', help="Collection")
parser.add_argument('--name', default='@jack', help="Meta name of memory")
args = parser.parse_args()

vdb = chromadb.PersistentClient(path=args.chroma_path)
memory = vdb.get_or_create_collection(name=args.collection)
memory.modify(f"{args.collection}-{ts}")

memory = vdb.get_or_create_collection(
    args.collection,
    metadata={
        "timestamp": ts,
        "name": args.name,
    },
)
