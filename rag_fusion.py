import os
import chromadb
from google import genai

# 1. Connect to the ChromaDB database you built in Step 2
db_client = chromadb.PersistentClient(path="./my_local_db")
collection = db_client.get_collection(name="my_first_rag")

# 2. Get the user's question dynamically from the terminal
print("-" * 40)
question = input("Ask a question about cricket, biryani, or C++: ")

# 3. Retrieve the single best matching context from our database
results = collection.query(
    query_texts=[question],
    n_results=1
)
retrieved_text = results["documents"][0][0]
print(f"\n[System] Found matching context from database:\n'{retrieved_text}'")

# 4. Connect to Gemini
# (Make sure GEMINI_API_KEY is exported in your terminal!)
llm_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 5. THE RAG PROMPT: This is the secret sauce. 
# We combine the retrieved text and the question into one giant string.
prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY the provided context. 
If the answer is not in the context, say "I don't know based on my data."

Context: {retrieved_text}

Question: {question}
"""

# 6. Generate the final answer
print("\n[System] Generating final answer with Gemini...\n")
response = llm_client.models.generate_content(
    model="gemini-3.6-flash", 
    contents=prompt
)

print("--- FINAL AI RESPONSE ---")
print(response.text)
print("-" * 40)
