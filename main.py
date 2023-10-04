from fastapi import FastAPI
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

app = FastAPI()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

class InputData(BaseModel):
    symptoms: str
    history: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['symptoms','history'],
        template = """AI, please consider yourself a doctor and provide a comprehensive response to the patient's symptoms, taking into account the patient's dignosis.
        Refer to the patient in first person and not the third.

        Patient's Current Symptoms:

        [Patient describes their current symptoms here.] only metion the symptoms given by the patient in one sentence don't add anything else.

        AI Doctor's Response:

        Based on the patient's reported symptoms and the provided dignosis, I recommend the following:

        Diagnosis:
        
        [Provide a potential diagnosis based on the reported symptoms and dignosis only if it is a contributing factor.]

        Treatment Recommendations:

        Medication: [Prescribe any necessary medications, dosage, and frequency.] be presise and not vague

        Lifestyle Modifications: [Recommend any lifestyle changes, such as dietary adjustments or exercise routines.]

        Follow-up: Schedule a follow-up appointment to assess the patient's progress and make any necessary adjustments to the treatment plan.

        Preventive Measures:

        [Provide advice on preventive measures or actions the patient can take to manage their condition and improve overall health.]

        Emergency Situations:

        [Explain under what circumstances the patient should seek immediate medical attention or contact emergency services.]

        Patient's Current Symptoms:
        {symptoms}
        Patient's Dignosis:
        {history}
        )
        AI Doctor's Response:
        """ 
    )

    llm = GooglePalm(temperature = 0.3)

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output')

    sequential_chain = SequentialChain(chains = [title_chain], input_variables = ['symptoms','history'], output_variables = ['output'], verbose = True)

    response = sequential_chain({'symptoms' : data.symptoms, 'history' : data.history})

    return response

