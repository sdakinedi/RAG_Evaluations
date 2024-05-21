import warnings
warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import fdm


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

#############RAG#############

############################ MANUAL ############################

questions = [
    "what is PartyEligibilityRegCalc"
]

ground_truth = [
    """
    PartyEligibilityRegCalc entity stores the eligibility of different parties for inclusion or exclusion for certain articles of certain regulations.
    Attributes of entity PartyEligibilityRegCalc:
    party : Description=>Party
    eligibilityType : Desctiption=>The article of part of regulation for which a certain exposure will be eligible for inclusion or exclusion.
    """
]


answers  = []
contexts = []


# traversing each question and passing into the chain to get answer from the system
for question in questions:
    answers.append(fdm.process_query(question))
    contexts.append([docs.content for docs in fdm.prepare_fdm_document_store(True).bm25_retrieval(question)])


llm = AzureChatOpenAI(
    openai_api_version="2023-08-01-preview",
    api_key="key",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-35-turbo-16k",
    #model=azure_configs["model_name"],
    #validate_base_url=False,
)
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-08-01-preview",
    api_key="key",
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="text-embedding-ada-002",
    #model=azure_configs["embedding_name"],
)


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
        #context_recall,
        #faithfulness,
        answer_relevancy,
    ],
    llm=llm,
    embeddings=azure_embeddings
)

df = result.to_pandas()
print(df)
df.to_html("ragas_eval_report6.html")

############################ SYNTHETIC ############################
'''
import ragas
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# generator with openai models
#generator = TestsetGenerator.with_openai(critic_llm = "gpt-4")
generator = TestsetGenerator.with_openai(critic_llm = "gpt-3.5-turbo")

#generator.adapt(language="whatever",evolutions=[simple, multi_context, reasoning])
#generator.save(evolutions=[simple, multi_context, reasoning])

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

