import chromadb

vdb = chromadb.PersistentClient(path="memory")
memory = vdb.get_or_create_collection("meta")

res = memory.get()
for i in range(len(res['documents'])):
    print("id: ", res['ids'][i])
    print("document: ", res['documents'][i])
    print("metadata: ", res['metadatas'][i])
    print('---------------')
