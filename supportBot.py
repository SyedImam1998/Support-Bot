from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

api_key  = os.getenv('GOOGLE_API_KEY')
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key,temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
vectordb_file_path = "faiss_index"

def createVectorDb():
    loader = CSVLoader(file_path="./qna.csv", source_column="Question")
    data = loader.load()
    db = FAISS.from_documents(data,embeddings)
    db.save_local(vectordb_file_path)
    
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    # score_threshold=0.7 will filtr the relevant vectors whos has similarity score close to 0.7
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "Please write a mail to syedimam1998@gmail.com" Don't try to make up an answer.
    Strictly I repeat donot make up answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriever,input_key="query",return_source_documents=True,chain_type_kwargs={"prompt":PROMPT})
    return chain

if __name__ == "__main__":
    # createVectorDb()
    chain = get_qa_chain()
    print(chain.invoke("Do you have python?"))

#If you execute the file independently (i.e., directly), then the code below the if __name__ == "__main__": block will get executed. If you import this file as a module into another script, the code below that block won't be executed automatically.    
    
    
    

