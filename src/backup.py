import chromadb
import json

vdb = chromadb.PersistentClient(path="../memory")
memory = vdb.get_or_create_collection(name="meta")

# Retrieve all memories
memories = memory.get(include=["documents", "metadatas"])

# Export memories to a file
with open("jack.bkp", "w") as f:
    json.dump(memories, f)