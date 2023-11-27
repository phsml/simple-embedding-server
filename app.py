import logging
from pathlib import Path

import tiktoken
from fastapi import FastAPI, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer


class Settings(BaseSettings):
    model: str

    model_config = SettingsConfigDict(env_file=".env.fastapi")


settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load embedding model
model_path = Path("./models") / settings.model.split("/")[-1]
if model_path.exists():
    logger.info("Start loading model")
    embeddings = SentenceTransformer(model_path, device="cuda")
    logger.info("Load model completed")
else:
    logger.info("Start downloading model")
    embeddings = SentenceTransformer(settings.model, device="cuda")
    embeddings.save(str(model_path))
    logger.info("Model saved")

app = FastAPI()


class RequestDataOpenAI(BaseModel):
    """Input data (OpenAI API format)"""

    input: list[list[int]]
    model: str
    encoding_format: str


class RequestDataLocalAI(BaseModel):
    """Input data (LocalAI format)"""

    input: list[str]
    model: str
    encoding_format: str


class OutputData(BaseModel):
    """Output data (OpenAI API format)"""

    data: list
    model: str
    object: str
    usage: dict


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/embeddings")
async def embedding_openai(request: Request, input_data: RequestDataOpenAI):
    """Perform similar processing to OpenAI API and return embeddings"""

    # Get input data
    payload = await request.json()
    input_data = RequestDataOpenAI(**payload)

    # Get tokenized input text
    encoded_inputs = input_data.input

    # Convert tokenized input sentences to text
    enc = tiktoken.get_encoding("cl100k_base")

    # [ToDo] Get the number of tokens
    n_tokens = len(encoded_inputs[0])

    # Create embeddings
    encoded_data = []
    for encoded_input in encoded_inputs:
        input_text = enc.decode(encoded_input)
        encoded_text = embeddings.encode(input_text).tolist()
        encoded_data.append(
            {
                "embedding": encoded_text,
                "index": 0,  # [ToDo] Confirm the need for the index field
                "object": "embedding",
            }
        )

    # Prepare output data
    output_data = OutputData(
        data=encoded_data,
        model=input_data.model,
        object="list",
        usage={"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    )

    return output_data


@app.post("/embeddings")
async def embedding_localai(request: Request, input_data: RequestDataLocalAI):
    """Perform similar processing to LocalAI and return embeddings"""

    # Get input data
    payload = await request.json()
    input_data = RequestDataLocalAI(**payload)

    # Get input text
    input_texts = input_data.input

    # Create embeddings
    encoded_data = []
    for input_text in input_texts:
        encoded_text = embeddings.encode(input_text).tolist()
        encoded_data.append(
            {
                "embedding": encoded_text,
                "index": 0,  # [ToDo] Confirm the need for the index field
                "object": "embedding",
            }
        )

    # Prepare output data
    output_data = OutputData(
        data=encoded_data,
        model=input_data.model,
        object="list",
        usage={
            "prompt_tokens": 100,  # dummy
            "total_tokens": 100,  # dummy
        },
    )

    return output_data
