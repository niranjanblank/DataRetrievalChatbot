import streamlit as st
from dotenv import load_dotenv
from utilities import get_qa_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Restaurant Chatbot")
    # keep the chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # setting the qa_chain if not stored in the session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_qa_chain()

    st.header("Restaurant Customer Service")
    st.caption("Please provide us your queries here.")

    #display the conversation
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # get the question from customer
    user_query = st.chat_input("Any Questions")

    # customer interaction
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        # save the message to chat history
        st.session_state.chat_history.append({"role":"user", "content":user_query})
        response = st.session_state.qa_chain({"question":user_query})
        # assistant message
        with st.chat_message("assistant"):
            st.write(response["answer"])

        st.session_state.chat_history.append({"role":"assistant","content":response["answer"]})
if __name__=="__main__":
    main()