import chromadb
from datetime import datetime, timezone

ts = int(datetime.now(timezone.utc).timestamp())

vdb = chromadb.PersistentClient(path="../memory")
memory = vdb.get_or_create_collection(name="meta")
memory.modify(f"meta-{ts}")

memory = vdb.get_or_create_collection(name="meta", metadata={
    "timestamp": ts,
    "name": "@jack",
})
