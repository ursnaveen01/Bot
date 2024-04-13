import os
import webbrowser
import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def read_csv_file(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain(query,cid):
    prompt_template = (
        f"{query} for Customer ID ={cid}. Give me answer in English.\n"
        "Anss\n\n"
        "Context:\n{context}?\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "You are good to go from beginning :)   Ask me Questions...."}]


def user_input(user_question,cid):
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(user_question,cid)

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Gemini CSV Chatbot",
        page_icon="🤖"
    )

    with st.sidebar:
        df1 = read_csv_file("bankingpaymentsFAQqueries.csv")
        df2 = read_csv_file("bankingpaymentsFAQ.csv")
        if df1 is not None and df2 is not None:
            all_text = df1.to_string(index=False) + '\n' + df2.to_string(index=False)
            text_chunks = get_text_chunks(all_text)
            get_vector_store(text_chunks)
            st.success(f"Welcome to Mangalyaan Chatbot")

    # Main content area for displaying chat messages
    st.title("Customer Support ChatBot🤖")
    st.write("Mangalyaan")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    if st.sidebar.button('Logout'):
        webbrowser.open_new_tab("LoginPage.html")  # Redirect to LoginPage.html


    # Chat input
    # Placeholder for chat messages
    def welcome_message(customer_name):
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
            {"role": "assistant", "content": f"Hi {customer_name}! Welcome to our Customer Support ChatBot for payments and banking! 🏦💳\n\nHow can we assist you today? Whether you have questions about transactions, account balances, or any other banking inquiries, feel free to ask. We're here to help you with any payment-related or banking-related concerns you may have.\n\nSimply type your query in the chatbox below, and we'll provide you with the assistance you need.\n\nLet's get started! 🚀"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

    # Display chat messages and bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt,customer_id)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)
    query_params = st.query_params
    customer_id = query_params.get("customerId", "")
    print(customer_id)

# Define a dictionary to map customer IDs to customer names
    customer_names = {
                "C3452": "shayan",
                "C3454":"abhishek",
                "C3456":"sharan",
                "C3458":"Naveen"
                
            };

# Check if the customerId exists in the dictionary
    if customer_id in customer_names:
        customer_name = customer_names[customer_id]
        st.write(welcome_message(customer_name))
    else:
    # If customerId doesn't exist, display a default welcome message
        st.write("Welcome! Please login to access the chatbot.")



if __name__ == "__main__":
    main()