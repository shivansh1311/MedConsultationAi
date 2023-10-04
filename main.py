
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as palm

app = FastAPI()

# Load the model and configure it
models = [
    m for m in palm.list_models() if "generateText" in m.supported_generation_methods
]

for m in models:
    print(f"Model Name: {m.name}")
model = models[0].name
model
palm.configure(api_key="AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic")

class InputData(BaseModel):
    text: str
    symptoms: str

class OutputData(BaseModel):
    result: str

@app.post("/generate")
async def generate_text(data: InputData):
    # Create a prompt using the provided text and symptoms
    
    prompt =f"""AI, please consider yourself a doctor and provide a comprehensive response to the patient's symptoms, taking into account the patient's medical history, lifestyle, and preferences.
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
    {data.symptoms}
    Patient's Medical History:
    {data.text}

    AI Doctor's Response:
    """ 
   
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=1000,
    )
    
    response = OutputData(result=completion.result)
    return response

