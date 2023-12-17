import pinecone
import os
os.environ["PINECONE_API_KEY"] = "64709212-3efc-426b-b031-e23b62938d05"
os.environ["PINECONE_ENV"] = "gcp-starter"

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)
pinecone.delete_index("frontera-poc")