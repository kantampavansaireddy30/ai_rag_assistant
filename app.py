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
        You are a helpful assistant. Answer the user's question using ONLY the provided context. 
        If the answer is not in the context, say "I don't know based on my data."

        Context: {retrieved_text}
        Question: {user_question}
        """
        
        with st.spinner("Thinking..."):
            response = llm_client.models.generate_content(
                model="gemini-3.6-flash", 
                contents=prompt
            )
            
            # Display the AI's answer and save it to history
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
