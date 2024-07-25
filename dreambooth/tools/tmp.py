# import nltk
# from nltk import Text
# from nltk.tokenize import RegexpTokenizer
# retokenize = RegexpTokenizer("[\w]+")
# emma_raw = nltk.corpus.gutenberg.raw("austen-emma.txt")
# text = Text(retokenize.tokenize(emma_raw))
# import nltk.corpus
# print(str(nltk.corpus.brown).replace('\\\\','/'))
# print(str(nltk.corpus.treebank).replace('\\\\','/'))
# from nltk.corpus import inaugural
# # inaugural.raw('1789-Washington.txt')
"""
a person is riding an orange motorcycle doing a hand motion ['motorcycle', 'person']
a person on a black and copper motorcycle. ['motorcycle', 'person']
guy riding his  gold motorcycle giving a signal. ['motorcycle', 'person']
a woman on an orange motorcycle with a helmet ['motorcycle', 'person']
person tiding an orange motorcycle with one had raised. ['motorcycle', 'person']
a person riding a motorcycle on a city street  ['motorcycle', 'person']
a man wearing a helmet rides his motorcycle down a street.  ['motorcycle', 'person']
a man riding a motorcycle down a street. ['motorcycle', 'person']
a person wearing a helmet is riding a deluxe motorcycle ['motorcycle', 'person']
a person riding a motorcycle on a roadway. ['motorcycle', 'person']
a subway train with a row of orange and white seats. ['train', 'bench']
three chairs under a window on a train. ['train', 'bench']
inside a subway with a view of the door windows and seats ['train', 'bench']
a full view of a train in a subway station. ['train', 'bench']
a seat and door on an empty train.  ['train', 'bench']
a bedroom with a large bed sitting under a painting. ['tv', 'chair', 'bed', 'cup', 'banana', 'dining table', 'vase', 'apple', 'orange']
a bedroom that has a door to the outside in it. ['tv', 'chair', 'bed', 'cup', 'banana', 'dining table', 'vase', 'apple', 'orange']
a couple of folded pillow cases are placed on the made bed.  ['tv', 'chair', 'bed', 'cup', 'banana', 'dining table', 'vase', 'apple', 'orange']
a white bed a green wall a television a chair and lamps ['tv', 'chair', 'bed', 'cup', 'banana', 'dining table', 'vase', 'apple', 'orange']
a green hotel room with a large bed set with two sets of pajamas.  ['tv', 'chair', 'bed', 'cup', 'banana', 'dining table', 'vase', 'apple', 'orange']
a large bedroom with big windows and a patio. ['tv', 'chair', 'bed']
a hotel room with a balcony and computer.  ['tv', 'chair', 'bed']
a spacious  bedroom with access to a balcony. ['tv', 'chair', 'bed']
a bed room with a bed and large clear doors ['tv', 'chair', 'bed']
a hotel room with a large bed and two giant windows. ['tv', 'chair', 'bed']
a small tv on the hard wood floor of a building ['tv', 'car']
a black tv is on the wood floor in an empty room. ['tv', 'car']
looking into a large window at a television in the corner on a wood floor. ['tv', 'car']
a tv in a room behind a window  ['tv', 'car']
a view from the outside of a window of a building  ['tv', 'car']
an airplane is parked on a runway and someone is standing nearby. ['airplane', 'person']
a airplane tha tis parked on a concrete runway ['airplane', 'person']
a man is walking by a plane on an airport field. ['airplane', 'person']
an aircraft that is on a airplane runway ['airplane', 'person']
a man standing on top of a tarmac next to a small airplane. ['airplane', 'person']
a small wheeled airplane on an open runway. ['airplane']
a small airplane is about to take off on the runway. ['airplane']
a small, single person aircraft sits on the runway. ['airplane']
a black plane taking off from an airport runway. ['airplane']
a grey and black plane on a runway with trees in the background. ['airplane']
a picture of a plane is flying in the air. ['airplane']
an airplane with two propellor engines flying in the sky. ['airplane']
a propeller plane flying through a blue sky. ['airplane']
a small airplane is flying through a clear blue sky. ['airplane']
a propeller plane flying through a blue sky. ['airplane']
a large jetliner sitting in front of a tall building. ['airplane']
a plane is on the tarmac at naha airport. ['airplane']
a red and white tail of a large plane ['airplane']
an airport with an airplane that has a red tale ['airplane']
an airplane is parked in front of a large building.  ['airplane']
a man driving a car across an airport runway. ['airplane', 'person', 'car', 'truck']
two airport workers transferring cargo from one plane to another ['airplane', 'person', 'car', 'truck']
people are transporting something to another area at the airport ['airplane', 'person', 'car', 'truck']
a worker driving a cart pulling a trailer loaded with cargo. ['airplane', 'person', 'car', 'truck']
"""
import spacy
nlp = spacy.load("en_core_web_lg")
# captions=[
#     "a worker driving a cart pulling a trailer loaded with cargo.",
#     "people are transporting something to another area at the airport",
#     "a man driving a car across an airport runway.",
#     "an airplane is parked in front of a large building.",
#           ]
captions=[
    "a bedroom that has a door to the outside in it.",
    "a couple of folded pillow cases are placed on the made bed.",
    "a white bed a green wall a television a chair and lamps",
    "a green hotel room with a large bed set with two sets of pajamas.",
    "a wii controller and cord is wrapped around a bottle of wine.",
    "two young children are playing together on a wii.",
    "two children in a room playing video games.",
 ]
# ['kitchen', 'outdoor', 'appliance', 'indoor', 'electronic', 'vehicle', 'food', 'accessory', 'sports', 'person', 'furniture', 'animal']
target_token=nlp('a furniture')[-1]
for caption in captions:
    tokens = nlp(caption)
    print()
    print(caption)
    for token in tokens:
        sim=token.similarity(target_token)
        # if token.pos_=='NOUN':
        if sim>0.4 and token.pos_=='NOUN':
            print(token.text,token.similarity(target_token),token.dep_)