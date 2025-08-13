import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
os.environ["HF_TOKEN"] = "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"

# Constants
DB_FAISS_PATH = "vectorstore"
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = ChatGroq(temperature=0.2, groq_api_key="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", model_name="llama3-8b-8192")

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

NON_CONTEXT_RESPONSES = {
    "greeting": "👋 Hello! How can I assist you with health-related queries today?",
    "default": "🤔 I'm here to help with health-related questions. Could you provide more context?",
}

def detect_non_contextual_prompt(prompt):
    """Detect if the user input is a non-contextual prompt."""
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if prompt.lower() in greetings:
        return "greeting"
    return None

import streamlit as st

def sidebar_chat_manager():
    """Handles chat session creation, switching, renaming, clearing, and deletion with three-dots menu."""
    st.sidebar.header("💬 Chat Sessions")

    # Initialize session states
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}  # {session_name: {"messages": [], "context": ""}}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None
    if "show_options" not in st.session_state:
        st.session_state.show_options = {}  # Tracks whether options are visible for each chat

    # Create a new chat
    if st.sidebar.button("➕ Start New Chat"):
        new_chat_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_chat_name] = {"messages": [], "context": ""}
        st.session_state.current_chat = new_chat_name
        st.session_state.show_options[new_chat_name] = False

    # Manage chats (switch, rename, clear, delete)
    for chat_name in list(st.session_state.chat_sessions):
        col1, col2 = st.sidebar.columns([0.8, 0.2])

        with col1:
            if st.button(f"📂 {chat_name}", key=f"open_chat_{chat_name}"):
                st.session_state.current_chat = chat_name
                st.session_state.show_options = {k: False for k in st.session_state.chat_sessions}

        with col2:
            if st.button("⋮", key=f"options_{chat_name}"):
                st.session_state.show_options[chat_name] = not st.session_state.show_options.get(chat_name, False)

        if st.session_state.show_options.get(chat_name, False):
            with st.sidebar:
                new_name = st.text_input(
                    label="Rename Chat",
                    value=chat_name,
                    key=f"rename_input_{chat_name}",
                    label_visibility="collapsed",
                )
                if new_name and new_name != chat_name:
                    if new_name in st.session_state.chat_sessions:
                        st.warning("Chat name already exists!")
                    else:
                        st.session_state.chat_sessions[new_name] = st.session_state.chat_sessions.pop(chat_name)
                        st.session_state.show_options[new_name] = st.session_state.show_options.pop(chat_name)
                        if st.session_state.current_chat == chat_name:
                            st.session_state.current_chat = new_name
                        st.rerun()  # Force UI update

                # Clear Chat Option
                if st.button("Clear Chat", key=f"clear_chat_{chat_name}"):
                    st.session_state.chat_sessions[chat_name]["messages"] = []  
                    st.session_state.chat_sessions[chat_name]["context"] = ""
                    if st.session_state.current_chat == chat_name:
                        st.session_state.current_chat = chat_name
                    st.success(f"Cleared all messages for {chat_name}.")

                # Delete Option
                if st.button("Delete Chat", key=f"delete_chat_{chat_name}"):
                    del st.session_state.chat_sessions[chat_name]
                    st.session_state.show_options.pop(chat_name, None)
                    if st.session_state.current_chat == chat_name:
                        st.session_state.current_chat = None
                    st.rerun()  # Force UI update

    # Show current chat name
    if st.session_state.current_chat:
        st.sidebar.markdown(f"**Current Chat:** {st.session_state.current_chat}")
# Load FAISS vectorstore
@st.cache_data
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Load LLM
def load_llm(huggingface_repo_id):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("HF_TOKEN is not set. Please check your environment variables.")
        return None
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"max_length": 512}
    )

# Set custom prompt template
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Main app
def main():
    
    st.title("🩺 Health Assistance Chatbot")
    st.markdown(
        """
        Welcome to the **Health Assistance Chatbot**! 🤖  
        Ask health-related questions, and I'll provide the best responses based on the given context.  
        You can manage multiple chats using the sidebar.
        """
    )

    # Sidebar management
    sidebar_chat_manager()

    # Check if a chat session is selected
    if st.session_state.current_chat is None:
        st.info("💡 Please create or select a chat session from the sidebar.")
        return

    current_chat_name = st.session_state.current_chat
    chat_data = st.session_state.chat_sessions[current_chat_name]

    # Display current chat header
    st.subheader(f"💬 Current Chat: {current_chat_name}")
    st.markdown("---")

    # Display existing messages
    for message in chat_data["messages"]:
        if message["role"] == "user":
            st.chat_message("user").markdown(f"🧑‍💻 **You:** {message['content']}")
        else:
            st.chat_message("assistant").markdown(f"🤖 **Assistant:** {message['content']}")

    # Load vectorstore
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.error("❌ Failed to load the vector store. Please check the database path.")
        return

    # Load LLM
    if llm is None:
        st.error("❌ Failed to load the LLM. Please check your HuggingFace token or model ID.")
        return

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # Chat input
    prompt = st.chat_input("Type your question here (e.g., 'What are the symptoms of diabetes?')")
    
    if prompt:
        # Display user message
        st.chat_message("user").markdown(f"🧑‍💻 **You:** {prompt}")
        chat_data["messages"].append({"role": "user", "content": prompt})

        # Detect non-contextual prompts
        non_context_key = detect_non_contextual_prompt(prompt)
        if non_context_key:
            # Handle non-contextual prompt
            response = NON_CONTEXT_RESPONSES[non_context_key]
            st.chat_message("assistant").markdown(f"🤖 **Assistant:** {response}")
            chat_data["messages"].append({"role": "assistant", "content": response})
        else:
            # Context-based response
            try:
                # Handle single-word or ambiguous queries
                if len(prompt.split()) < 2:  # Detect short or ambiguous queries
                    refined_prompt = f"What do you want to know about {prompt.strip()}? For example, 'What is {prompt.strip()}?' or 'Symptoms of {prompt.strip()}'."
                    st.chat_message("assistant").markdown(f"🤖 **Assistant:** {refined_prompt}")
                    chat_data["messages"].append({"role": "assistant", "content": refined_prompt})
                else:
                    if prompt.lower().startswith("context:"):
                        # Extract new context and don't append old context
                        new_context = prompt.replace("Context:", "").strip()
                        full_context = new_context
                        chat_data["context"] = new_context  # Replace old context with new one
                    else:
                        if chat_data["context"]:
                            # Use old context if available
                            full_context = f"{chat_data['context']}\n{prompt}"
                            clarification_message = (
                                f"I apolgize, I am unable to fully understand what your context is, but based on previous chats, is this what you want[if not, can you rephrase and clarify your question]?\n\n"
                            )
                        else:
                            full_context = prompt
                            clarification_message = ("")

                    # Generate the response
                    response = qa_chain({"query": full_context})
                    result = response.get("result", "No response available.")
                    source_documents = response.get("source_documents", [])

                    # Combine clarification message and response
                    combined_response = clarification_message + result

                    # Format and display source documents
                    with st.expander("📄 View Source Documents"):
                        for idx, doc in enumerate(source_documents):
                            st.markdown(f"**Document {idx + 1}:** {doc.page_content[:200]}...")

                    # Update chat context if no explicit context reset
                    if not prompt.lower().startswith("context:"):
                        chat_data["context"] = result

                    # Display and save response
                    st.chat_message("assistant").markdown(combined_response)
                    chat_data["messages"].append({"role": "assistant", "content": combined_response})

            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()