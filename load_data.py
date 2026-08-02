import chromadb

# 1. Connect to the exact same local database folder your Streamlit app uses
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Connect to the same collection
collection = client.get_or_create_collection(name="rag_collection")

# 3. Add your friends' data
collection.add(
    documents=[
        "My friend Alex is a software engineer who loves playing cricket and eating biryani.",
        "Sarah is a graphic designer and she usually hangs out at the local coffee shop.",
        "Jordan's birthday is on October 12th and their favorite movie is The Matrix."
    ],
    metadatas=[
        {"category": "friend", "name": "Alex"},
        {"category": "friend", "name": "Sarah"},
        {"category": "friend", "name": "Jordan"}
    ],
    ids=["friend_001", "friend_002", "friend_003"] # These must be unique strings!
)

print("Friends data successfully added to ChromaDB!")
