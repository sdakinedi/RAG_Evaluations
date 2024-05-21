import warnings
warnings.filterwarnings('ignore')
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
#import rag.fdm_v2 as fdm2
#import rag.fdm_v1 as fdm1
import rag.rag_chain as fdm
from rag.rag_chain import RagChain
from search.hybrid_search import HybridSearchRetriever
from services.rag_service import RagService
#import utils.fdm_data as fdm_data
import dataload.fdm_data as fdm_data
#import fdm_advanced
from utils.langchain_helpers import convert_text_units_to_lc_documents, formatted_text, merge_up, remove_repeated_documents, limit_by_size, pretty_print_documents_headers
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
import os
from dotenv import load_dotenv
# Load the .env file into the environment variables
load_dotenv()
############################ RAGAs  ############################ 
############################ MANUAL TEST DATA SETTING ############################

#OUT OF SCOPE NOW
'''
#Evaluating v1
def evaluating_agentic(llm, azure_embeddings, questions, ground_truth):
    answers  = []
    contexts = []
    text_units = fdm_data.load_input_data()
    # traversing each question and passing into the chain to get answer from the system
    for question in questions:
        #answer, documents = fdm.get_answer_and_documents(question)
        #context = [document.metadata["text"] for document in documents]
        #answers.append(answer)
        #contexts.append(context)
        answers.append(fdm1.process_query(question))
        contexts.append([docs.content for docs in fdm1.prepare_fdm_document_store(text_units=text_units,in_memory=True).bm25_retrieval(question)])

    # Preparing the dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth
    }

    df = preparing_evaluation(data,llm,azure_embeddings)
    df.to_html("ragas_eval_reportv1.html")
'''  
#Evaluating version v2
def evaluating_v2(llm, azure_embeddings, questions, ground_truth):
    answers  = []
    contexts = []
    #fdm.initialize()
    rag_service = RagService()    
    for question in questions:
        answer, documents = rag_service.get_answer_and_documents_with_rag_chain(question)
        context = [document.metadata["text"] for document in documents]
        answers.append(answer)
        contexts.append(context)
        #answers.append(fdm_advanced.process_query(question))
        #contexts.append([docs.page_content for docs in fdm_advanced.get_relevant_documents(question)])
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth
    }

    df = preparing_evaluation(data,llm,azure_embeddings)
    df.to_html("ragas_eval_reportv2.html")
    
#Evaluating metric
def preparing_evaluation(data,llm,azure_embeddings):    
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
        llm=llm,
        embeddings=azure_embeddings
    )
    df = result.to_pandas()
    return df

def preparing_llm_and_setup():
    llm = AzureChatOpenAI(
    openai_api_version="2023-08-01-preview",
    api_key=<key_to_enter>,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-35-turbo-16k",
    #model=azure_configs["model_name"],
    #validate_base_url=False,
    )
    azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-08-01-preview",
    api_key=<key_to_enter>,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="text-embedding-ada-002",
    #model=azure_configs["embedding_name"],
    )
    
    #Questions to ask
    questions = [
        "What is the 'eligibilityType' attribute in the 'CommitmentEligibilityRegCalc' entity?",
        "What are O-SII and G-SII?",
        "F2093000' code in the document?",
        "Difference between F1023413 and F1023414 codes?",
        "What does the suffix 'Rate' imply in attribute naming conventions?",
        "entity InternalParty is inherited from which entity?",
        "List of attributes for entity PartyLink",
        "list all the logical column name & data type of attributes for entity party without any other details like description, length, inherited form, etc.",
        "What does the 'yearToDateLocalCurrencyRestated' attribute represent?",
        "the purpose of the entity 'ResponsibilityCentre' and what are its attributes",
    ]
    
    #Ground Truths against the questions
    ground_truth = [
    """
        The 'eligibilityType' attribute is the article of part of regulation for which a certain exposure will be 
        eligible for inclusion or exclusion.The attribute if of string data type, having length of 150 and is inherited from EligibilityRegCalc
    """,
    """
        O-SII stands for Other Systematically Important Institutions and G-SII stands for Global Systematically Important Institutions. 
        These are financial institutions whose distress or disorderly failure, because of their size, complexity and systemic interconnection, would cause significant disruption to the wider financial system and economic activity""",
    """
        The 'F2093000' code represents Provisions -- Restructuring
    """,
    """
        F1023413 represents ""Financial Assets -- Designated at Fair Value -- Loans and advances -- Interest rate -- Trade receivables. F1023414 represents ""Financial Assets -- Designated at Fair Value -- Loans and advances -- Interest rate -- Finance leases.In summary, F1023413 is used for trade receivables whereas F1023414 is used for financial leases.
    """,
    """
        Suffix 'Rate' implies a periodicity (for example, This field represents a fee rate and it needs to be provided if the fee is expressed in function of the linked Position or Commitment. 
        For Securities Lending, for instance, the rate will apply to the market value of the loaned securities, continuously compounding rate) 
        and refers to some kind of market/interest rates.
        """,
        """
            The entity InternalParty is inherited from the entity LegalParty.
        """,
        """
        The entity PartyLink has the following attributes:

            party1: This attribute represents a party involved in the relationship. The data type for this attribute is 'Party'.

            party2: This attribute represents another party involved in the relationship. The data type for this attribute is also 'Party'.

            partyLinkType: This attribute represents the type of relationship link between a party and its parent entity. The possible types can be first parent, ultimate parent, headoffice, etc. The data type for this attribute is 'string' with a length of 20.
        """,
        """
        The logical column names and their corresponding data types for the entity Party are as follows:

            partyKey: string
            description: string
            partyType: string
            countryOfLegalDomicileType: string
            legalProceedingStartDate: date
            legalProceedingType: string
            nuts3Type: string
            streetName: string
            streetNumber: string
            city: string
            postalCode: string
            pdPartyValuation: PartyValuation 
            
        """,
        """
            The 'yearToDateLocalCurrencyRestated' attribute represents the amount associated to a balance referring to a specific year 
            and expressed in the local currency of the internal party, as derived from the restatement process.
            It has decimal as data type and 28,8 as length.
        """,
        """
            ResponsibilityCentre entity will hold the responsibility centres that make up the internal parties.
            The attributes of the "ResponsibilityCentre" entity are as follows:
            1. responsibilityCentreKey: The unique identifier of a responsibility center.Data Type is string with length as 50.

            2. internalParty: This attribute represents the internal party to which the responsibility center belongs.Data type is InternalParty.

            3. description: This attribute provides a descriptive representation of entities.Data type is string with length as 800.

            4. responsibilityCentreType: This attribute indicates the type of responsibility center.Data type is string with length as 10.
        """,
    ]
    
    #evaluating_v1(llm, azure_embeddings, questions, ground_truth) 
    evaluating_v2(llm, azure_embeddings, questions, ground_truth)
    
if __name__ == '__main__': 
    preparing_llm_and_setup()
    
    
############################ SYNTHETIC TEST DATA : Work in progress ############################
'''
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(
    generator_llm=llm, critic_llm=llm, embeddings=azure_embeddings
)
#documents = [docs for docs in fdm.prepare_fdm_document_store(True).filter_documents()]
full_text_units = fdm_data.get_full_text_units()
#store = fdm.prepare_fdm_document_store(text_units=text_units,in_memory=True)
documents = convert_text_units_to_lc_documents(full_text_units)
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=1,
    raise_exceptions=False,
    with_debugging_logs=False,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    is_async=False,
)

df_testset = testset.to_pandas()
print(df_testset)
'''


