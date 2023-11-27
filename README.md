# Simple embedding server
## Usage
### Build
```bash
cp .env.sample .env
docker compose up --build
```

### Example
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    # model="text-embedding-ada-002",  # dummy model
    openai_api_base=base_url,
    openai_api_key="sk-",  # dummy
)
query_result = embeddings.embed_query("This is a test document.")
```
