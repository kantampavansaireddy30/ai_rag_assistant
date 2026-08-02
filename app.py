import os
import streamlit as st
import chromadb
from google import genai

# 1. UI Configuration
st.set_page_config(page_title="My RAG Assistant", page_icon="🤖")
st.title("My First RAG App 🚀")

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
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about cricket, biryani, or C++!"}]

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
                # 1. TRY to call the API
                response = llm_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                # 2. If it works, write the answer to the screen
                st.write(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                # (Make sure you also append the response to your session_state history here!)
                
            except Exception as e:
                # 3. If it FAILS, catch the error and show a polite UI message
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning("⏳ We hit our API quota limit! If you've asked a lot of questions today, you may need to wait until tomorrow for the daily reset.")
                else:
                    st.error(f"An API error occurred: {e}")
