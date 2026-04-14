from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Create database
db = client["ai_recommender"]

# Create collection
collection = db["products"]

# Sample products (same as your products.py)
products = [
    {
        "name": "iPhone 16",
        "description": "Great camera, smooth performance, premium design",
        "category": "phone",
        "price": 60000
    },
    {
        "name": "Samsung Galaxy S25",
        "description": "Excellent display, strong performance, good camera",
        "category": "phone",
        "price": 50000
    },
    {
        "name": "OnePlus 15",
        "description": "Fast performance, gaming friendly, smooth UI",
        "category": "phone",
        "price": 55000
    },
    {
        "name": "MacBook Air M4",
        "description": "Lightweight laptop, great battery life",
        "category": "laptop",
        "price": 80000
    }
]

# Insert into DB
collection.insert_many(products)

print("✅ Products inserted into MongoDB")