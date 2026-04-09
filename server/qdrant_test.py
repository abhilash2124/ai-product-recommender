# sourcery skip: for-append-to-extend, list-comprehension
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Sentences
sentences = [
    "I love coding",
    "I enjoy programming",
    "I hate bugs",
    "abhilash is a developer",
    "abhilash likes programming+"
]

# Convert to embeddings
embeddings = model.encode(sentences)

# Create Qdrant client
client = QdrantClient(":memory:")

# Create collection
if not client.collection_exists("my_collection"):
    client.create_collection(
        collection_name="my_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Prepare points
points = []
for i, vector in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={"text": sentences[i]}
        )
    )

# Store data
client.upsert(
    collection_name="my_collection",
    points=points
)

print("✅ Data stored in Qdrant!")

# Querying
query = "coding is fun"

query_vector = model.encode(query)

result = client.query_points(
    collection_name="my_collection",
    query=query_vector,
    limit=2
)

print("\n 🔍 Search results:")
for res in result.points:
    print(res.payload["text"], " | Score:", res.score)
    


# Step 1: Get top results from Qdrant
context = "\n".join([res.payload["text"] for res in result.points])

# Step 2: Create prompt
prompt = f"""
### ROLE
You are a highly accurate Knowledge Assistant. Your goal is to answer questions based strictly on the provided Context.

### CONSTRAINTS
- Use ONLY the provided Context to answer.
- If the answer is not contained within the Context, explicitly state: "I'm sorry, I don't have information about that in my current database."
- Do not use any outside knowledge or "hallucinate" facts.
- Keep the tone professional, concise, and helpful.

### CONTEXT
{context}

### USER QUESTION
{query}

### FINAL ANSWER
"""

# Step 3: Call LLM
response = client_groq.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Step 4: Print answer
print("\n🤖 AI Answer:")
print(response.choices[0].message.content)