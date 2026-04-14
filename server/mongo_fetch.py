from pymongo import MongoClient

# Connect
client = MongoClient("mongodb://localhost:27017/")
db = client["ai_recommender"]
collection = db["products"]

# Fetch all products
products = list(collection.find({}, {"_id": 0}))

# Print
for p in products:
    print(p)