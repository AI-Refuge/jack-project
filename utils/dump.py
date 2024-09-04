import chromadb

vdb = chromadb.PersistentClient(path="../memory")

print('-----meta------')
memory = vdb.get_or_create_collection("meta")
res = memory.get()
for i in range(len(res['documents'])):
    print("id: ", res['ids'][i])
    print("document: ", res['documents'][i])
    print("metadata: ", res['metadatas'][i])
    print('---------------')

print('--conv-first---')
conv = vdb.get_or_create_collection("conv-first")
res = conv.get()
for i in range(len(res['documents'])):
    print("id: ", res['ids'][i])
    print("document: ", res['documents'][i])
    print("metadata: ", res['metadatas'][i])
    print('---------------')
