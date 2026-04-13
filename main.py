from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 4, "fetch_k": 10, "lambda_mult": 0.5} #Lambda_mult controls the balance between relevance and diversity in the retrieved results. A value of 0.5 means that both relevance and diversity are equally weighted.
)

llm = ChatOpenAI(model="gpt-4.1-nano")

prompt = ChatPromptTemplate.from_messages(
[
    ("system", 
     """You are a helpful assistant.
Use ONLY the provided context to answer the question. 
If you don't know the answer, say you don't know. 
Do not use any information that is not provided in the context.
"""
   ),
    MessagesPlaceholder(variable_name="chat_history"),
   ("human",
     """Context: 
        {context}
        Question: 
        {question}
        """) 
])

chat_history = []

while True: 
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting the program.")
        break

    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    response = prompt.invoke({"context": context, "question": query, "chat_history": chat_history})
    answer = llm.invoke(response)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer.content))
    
    print(f"AI: {answer.content}\n")


