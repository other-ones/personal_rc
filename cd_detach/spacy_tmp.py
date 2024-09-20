import time
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample string
# text = "a girl is crying."
text = "a dog on top of ivory fan"
doc = nlp(text)
for token in doc:
    print(f"{token.text}: {token.pos_}")
# words=text.split()
# st=time.time()
# for _ in range(1000):
#     for word in words:
#         # Process the text
#         doc = nlp(text)
#         print(doc[0].pos_)
#         exit()
#         # Print each word with its corresponding part of speech
        
# print((time.time()-st)/1000)