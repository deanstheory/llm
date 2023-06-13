# https://github.com/logspace-ai/langflow

import os
from mylangflow import load_flow_from_json

os.environ["OPENAI_API_KEY"] = "my_key"

flow = load_flow_from_json("./langflow_pdf_loader.json")
# Now you can use it like any chain
flow("Hey, have you heard of LangFlow?")

