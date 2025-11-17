import spacy
import re

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I love the new iPhone 15. It's amazing!",
    "The Samsung Galaxy S23 is great, but the battery life is poor.",
    "I bought a Dell laptop, and it works perfectly.",
    "Amazon Echo is a good product, but the voice recognition is not reliable.",
    "The product is not as described. Very disappointed with the brand."
]

# Rule-based sentiment analysis function
def rule_based_sentiment(text):
    positive_words = ["love", "amazing", "great", "perfect", "good", "excellent"]
    negative_words = ["not", "poor", "disappointed", "bad", "unreliable", "not reliable"]

    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())

    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"

# Process each review
for i, review in enumerate(reviews):
    doc = nlp(review)

    # Extract product names and brands using NER
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]

    # Extract product names using regex (for brand names like "iPhone", "Samsung", etc.)
    product_brands = re.findall(r'\b(?:iPhone|Samsung|Dell|Amazon|Echo|Galaxy|S23|15)\b', review, re.IGNORECASE)

    # Perform rule-based sentiment analysis
    sentiment = rule_based_sentiment(review)

    # Print results
    print(f"Review {i+1}:")
    print(f"  Text: {review}")
    print(f"  Extracted Entities (NER): {entities}")
    print(f"  Extracted Brands/Products (Regex): {product_brands}")
    print(f"  Sentiment: {sentiment}")
    print("-" * 50)