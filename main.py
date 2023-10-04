from fastapi import FastAPI, HTTPException
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
        template = """AI, please consider yourself a doctor and provide a comprehensive response to the patient's symptoms, taking into account the patient's medical history, lifestyle, and preferences.
        Refer to the patient in first person and not the third.

        Patient's Current Symptoms:

        [Patient describes their current symptoms here.] only metion the symptoms given by the patient in one sentence don't add anything else.

        AI Doctor's Response:

        Based on the patient's reported symptoms and the provided medical history, I recommend the following:

        Diagnosis:
        Mention what from his medical history could be contrubuting to this sympton.

        [Provide a potential diagnosis based on the reported symptoms and medical history.]

        Treatment Recommendations:

        Medication: [Prescribe any necessary medications, dosage, and frequency.]

        Lifestyle Modifications: [Recommend any lifestyle changes, such as dietary adjustments or exercise routines.]

        Follow-up: Schedule a follow-up appointment to assess the patient's progress and make any necessary adjustments to the treatment plan.

        Preventive Measures:

        [Provide advice on preventive measures or actions the patient can take to manage their condition and improve overall health.]

        Emergency Situations:

        [Explain under what circumstances the patient should seek immediate medical attention or contact emergency services.]

        Patient's Current Symptoms:
        {symptoms}
        Patient's Medical History:
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
    
 except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
