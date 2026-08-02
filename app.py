import os
import streamlit as st
import sys

# 1. SQLite Patch for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from google import genai

# 2. Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_collection")

# 3. Initialize Google Gemini Client
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.write("Welcome to my RAG Assistant. Ask me anything about my data!")



#0 sidebar
with st.sidebar:
    st.title("⚙️ App Controls")
    st.write("Welcome to my custom AI application.")
    st.divider() # Draws a neat visual line
    
    # A functional button to wipe the memory clean
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Refreshes the UI instantly

# 1. UI Configuration
st.set_page_config(
    page_title="Pavan's RAG Assistant",
    page_icon="🚀",
    layout="centered" # Change to "wide" if you want the app to span the whole screen
)

# 2. Database Connection
# @st.cache_resource prevents Streamlit from reconnecting to the DB every time you chat
@st.cache_resource 
def get_db_collection():
    db_client = chromadb.PersistentClient(path="./my_local_db")
    return db_client.get_collection(name="my_first_rag")

collection = get_db_collection()
llm_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me about my friends with their names"}]

# 4. Render previous messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. Chat Input Box
if user_question := st.chat_input("Ask a question..."):
    
    # Immediately display the user's question in the UI
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
        
    # RAG Retrieval Process
    with st.chat_message("assistant"):
        with st.spinner("Searching database..."):
            results = collection.query(
                query_texts=[user_question],
                n_results=1
            )
            retrieved_text = results["documents"][0][0]
            
        # Generate Prompt
        prompt = f"""
        You are a helpful and intelligent AI assistant. 
        
        First, evaluate if the user's question can be answered using the following Context retrieved from our local database:
        Context: {retrieved_text}
        
        If the Context contains relevant information, use it to answer the question accurately. 
        If the Context is completely unrelated or missing the answer, ignore it and answer the user's question using your own general knowledge. Do not mention that the context was missing; just answer the question naturally.

        Question: {user_question}
        """
        
        with st.spinner("Thinking..."):
            try:
                # --- THE RAG MAGIC STARTS HERE ---
                
                # 1. Search the database using the user's question
                db_results = collection.query(
                    query_texts=[prompt],
                    n_results=1 # Pull the top 1 most relevant fact
                )
                
                # 2. Extract the text facts from the database results
                retrieved_facts = db_results["documents"][0]
                context_string = "\n".join(retrieved_facts)
                
                # 3. Create a secret super-prompt for Gemini
                augmented_prompt = f"""
                You are a helpful assistant answering questions about my friends.
                Use ONLY the following facts to answer the user's question. 
                If the answer is not in the facts, say "I don't have that information in my database."
                
                Database Facts:
                {context_string}
                
                User Question:
                {prompt}
                """
                
                # 4. Send the secret augmented prompt using gemini_client!
                response = gemini_client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=augmented_prompt
                )
            
                
                # --- THE RAG MAGIC ENDS HERE ---
                
                st.write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                # 3. If it FAILS, catch the error and show a polite UI message
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("⏳ We hit our API quota limit! If you've asked a lot of questions today, you may need to wait until tomorrow for the daily reset.")
                else:
                    st.error(f"An API error occurred: {e}")
