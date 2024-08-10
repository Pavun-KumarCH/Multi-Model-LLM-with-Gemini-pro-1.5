import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import secrets

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program.multi_modal_llm_program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import SimpleDirectoryReader

from decouple import config
from pydantic import BaseModel


load_dotenv(find_dotenv())


# Alternative 
GOOGLE_API_KEY = config("GOOGLE_API_KEY")
Model_name = "models/gemini-1.5-flash-latest"

# Initialize the Class for personal attributes
class PersonAttributes(BaseModel):
    """Data model of Description of person"""
    name: str
    nationality: str
    date_of_birth: str
    place_of_birth: str
    latittude_of_place_of_birth: float
    longitude_of_place_of_birth: float
    height: float
    weight_in_kilograms: float



# Create a Prompt Template
prompt_template  = """\
    Given me a Summary of the Person in the Image\
        and return your response with a json format\
        """

def structured_response_gemini(
        output_class: PersonAttributes,
        image_documents: list,
        prompt_template: str,
        model_name: str = Model_name
        ):
    # Initialize the Gemini MultiModal LLM
        gemini_llms = GeminiMultiModal(
              api_key = GOOGLE_API_KEY,
              model_name = model_name,
        )

        # Create a llm-completion Program
        llm_program = MultiModalLLMCompletionProgram.from_defaults(
              output_parser = PydanticOutputParser(output_cls = output_class),
              image_documents = image_documents,
              prompt_template_str = prompt_template,
              multi_model_llm = gemini_llms,
              verbose = True
        )

        response = llm_program()

        return response.result

def get_details_from_multimodel_gemini(uploaded_image):
      """get response
      Process uploaded image and fetch details from Gemini MultiModal LLM
      """
      # Load image documents
      for image_doc in uploaded_image:
            data_list = []
            structured_response = structured_response_gemini(
                  output_class=PersonAttributes,
                  image_documents =[image_doc],
                  prompt_template = prompt_template,
                  model_name = Model_name
            )
      # Get structured response from Gemini MultiModal LLM
            for r in structured_response:
                 data_list.append(r)

            data_dict = dict(data_list)
            return data_dict

uploaded_file = st.file_uploader(
      "Choose An Image File",
      accept_multiple_files = False,
      type = ["png", "jpg"]
)

if uploaded_file is not None:
      st.toast("File uploaded sucessfully")
      byte_data = uploaded_file.read()
      st.write("Filename: ", uploaded_file.name)

      with st.spinner("Loading, please wait"):
            if uploaded_file.type == "images/jpeg":
                  file_type = "jpg"
            else:
                  file_type = "png"
        
            # save file
            filename = f"{secrets.token_hex(8)}.{file_type}"

            with open(f"./images/{filename}", "wb") as fp:
                  fp.write(byte_data)
            
            file_path = f"./images/{filename}"


            #load images
            image_documents = SimpleDirectoryReader(input_files = [file_path]).load_data()

            response = get_details_from_multimodel_gemini(uploaded_image = image_documents)
            
            with st.sidebar:
                  st.image(image = file_path, caption = response.get("name", "Unknown"))
                  st.markdown(f"""
                              :green[Name]: :red[{response.get("name", "Unknown")}]\n
                              :green[Natinality]: :violet[{response.get("nationality", "Unknown")}]\n
                              :green[Date Of Birth]: :gray[{response.get("date_of_birth", "Unknown")}]\n
                              :green[Place of Birth]: :orange[{response.get("place_of_birth", "Unknown")}]\n
                              :green[Height]: :red[{response.get("height", "Unknown")}]\n
                              :green[Weight In Kilogram]: :red[{response.get("weight_in_kilograms", "Unkown")}]\n
                            """
                            )
                  
            df = pd.DataFrame({"Latitude" : response.get("latitude_of_place_of_birth", 0.0),
                               "Longitude" : response.get("longitude_of_place_of_birth", 0)},
                               index = [0]
                             )
            
            st.map(df, latitude = "Latitude", longitude = "Longitude")