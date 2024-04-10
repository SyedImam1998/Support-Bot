import streamlit as st
from supportBot import createVectorDb,get_qa_chain
st.title("Welcome to Support Bot 🤖")

latest_data=st.button("Get Latest Data")
question=st.text_input(label="Ask Question",placeholder="Please enter you question here....")

if latest_data:
    createVectorDb()

if question:
    chain=get_qa_chain()
    response=chain.invoke(question)
    st.header("Answer:")
    st.write(response['result'])