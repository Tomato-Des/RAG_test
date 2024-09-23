import os
from dotenv import load_dotenv #change this line to your own GROQ API KEY,GROQ_API_KEY=<Your Key>
from groq import Groq
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

#Load enviroment variables from .env file
load_dotenv()
class SentenceTransformerEmbeddings(Embeddings):
#FAISS.from_documents call the function embed_documents
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)
    
    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=False)
# Initialize model
#model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

# Load documents from the 'Data' directory
loader = DirectoryLoader('Data', glob="**/*.txt")
docs = loader.load()

# Split the documents for better retrieval
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Create a FAISS vector store from embedded documents
#vector = FAISS.from_documents(documents, embedding_model) #create faiss vector store
#vector.save_local('path to store your faiss vector store') # save to local
vector = FAISS.load_local('path of your stored faiss vector store', embedding_model, allow_dangerous_deserialization=True) #load from local

# Initialize the language model for the retrieval
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# Create the prompt template for the RAG process
prompt_template = """
If you don't have enough information to answer a question confidently, state that you don't have sufficient information to provide a reliable answer.
Avoid making assumptions or extrapolating beyond the data you have.
If a question is ambiguous, ask for clarification instead of guessing the intent.
Don't generate or invent information to fill gaps in your knowledge.
If you're unsure about any part of your answer, explicitly state your uncertainty.
    Prioritize answering the question based on the provided context:

    <context>
    {context}
    </context>
    Do not try
    Question: {input}"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Create a chain that combines the documents and language model
document_chain = create_stuff_documents_chain(llm, prompt)

# Convert the vector store into a retriever
retriever = vector.as_retriever()

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to run the RAG process with user input
def ask_question():
    print("You can start asking questions (type 'exitt' to quit):")
    while True:
        # Take user input
        user_input = input("\nEnter your question: ")

        # Check if the user wants to exit
        if user_input.lower() == 'exitt':
            print("Exiting...")
            break

        # Run the RAG process
        result = retrieval_chain.invoke({"input": user_input}) #output only 'answer' key
        """
        keys in the result dictionary: 
        - input
        - context
        - answer
        (verified by print)
        print("\nkeys in the result dictionary:")
        for key in result.keys():
            print(f"- {key}")
        """
        answer = result['answer'] #print only the answer part

        # Print the generated answer
        print("\nAnswer:", answer)
        

# Start the question-answering loop
if __name__ == "__main__":
    ask_question()

