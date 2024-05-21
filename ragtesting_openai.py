import warnings
warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv,find_dotenv
# Load OpenAI API key from .env file
load_dotenv(find_dotenv())

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Loading the samle data using the TextLoader
pages = []
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./state_of_the_union.txt")
# loader.load_and_split()
local_pages = loader.load_and_split() # load and split the pdf into pages
print(len(local_pages))
pages.extend(local_pages)

# split the pages into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(pages)

# Create a vector store for the sample data
persist_directory = 'data/chroma/'
#!rm -rf ./data/chroma  # remove old database files if any

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()

retriever = vectordb.as_retriever()

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define prompt template
prompt_template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)


retrieval = RunnableParallel(
    {"context": retriever,  "question": RunnablePassthrough()} 
)

chain = retrieval | prompt | llm | StrOutputParser()



#############RAG#############

############################ MANUAL ############################

questions = [
    "What did Putin do few days ago?"
]
ground_truth = [
    "Six days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. "
]

answers  = []
contexts = []

# traversing each question and passing into the chain to get answer from the system
for question in questions:
    answers.append(chain.invoke(question))
    print(f"answer is {answers[0]}")
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])

# Preparing the dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
print(df)
df.to_html("ragas_eval_report4.html")

############################ SYNTHETIC ############################
'''
import ragas
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# generator with openai models
generator = TestsetGenerator.with_openai(critic_llm = "gpt-4")

# Change resulting question type distribution
distributions = {
    simple: 0.3,
    multi_context: 0.2,
    reasoning: 0.5
}


testset = generator.generate_with_langchain_docs(pages, 1, distributions) 
df = testset.to_pandas()
print(df)
'''