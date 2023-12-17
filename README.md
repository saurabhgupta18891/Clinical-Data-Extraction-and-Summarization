

# Clinical-Data-Extraction-and-Summarization


This Project is designed to provide medical information by answering user queries using state-of-the-art language models and vector stores ,there are 3 individual tasks NER,Question answer and summarization which this project do . This README will guide you through the setup and usage of the Clinical Data Extraction and Summarization.

## Table of Contents

- [Introduction](#langchain-medical-bot)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

make sure you have the following prerequisites installed on your system.

- Python 3.9 or higher
- Required Python packages (you can install them using pip):
    - langchain
    - pinecone-client
    - sentence-transformers
    - transformers
    - tiktoken
    - huggingface_hub
    - openai==0.28
    - docx2txt
    - flask
    - datasets

## Installation

1. Create a Python virtual environment or conda environment:

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Use open ai api key which is provided in the API script itself

5. Use pinecone api key and environment,that is mentioned in the script.

## Getting Started

To get started with the Project, you need to:

1. Set up your environment and install the required packages as described in the Installation section.

2. Start the project by running the provided Python script(API.py).

3. Install Postman to test the flask API using the request(Final_Result image is given).


## Usage

The Project can be used for answering medical-related queries,extracting medical entities and generating a summary. To use the project, you can follow these steps:

1. Start the application by running  the provided Python script(API.py).

2. Send a medical-related query and required medical document(.docx format only) as a request through postman.

3. The app will provide an answer to input query based on information available in the document and will return medical entities and summary in one single response as json(please check Final_Result.png file) .

4. the app will generate summary by using a custom pretrained(finetuned) model(Saurabh91/medical_summarization-finetuned-starmpccAsclepius-Synthetic-Clinical-Notes) which I trained on clinical notes data.

5. jupyter notebook to finetune a model for summarization is provided in the project folder.

##Important Notes

1.The API will give response in a few minutes due to processing of three modules simultaneously.

2.I also did r&d on Llama-2 quantized model for inferencing(results are not that good as GPT-3.5)

3.please delete pinecone index before generating a new request with new document(limitation of pinecone free version.only one index for one document can be created),(run pinecone_delete.py file and then generate a new api request)

 🚀
