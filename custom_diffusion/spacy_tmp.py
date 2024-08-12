import time
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample string
# text = "a girl is crying."
text = "a dog holding a paper saying 'red apple'"
words=text.split()
st=time.time()
for _ in range(1000):
    for word in words:
        # Process the text
        doc = nlp(text)
        # Print each word with its corresponding part of speech
        # for token in doc:
        #     print(f"{token.text}: {token.pos_}")
print((time.time()-st)/1000)