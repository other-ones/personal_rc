from spacy import symbols
import spacy

nlp = spacy.load("en_core_web_sm")
# # pos_tags = nlp.get_pipe("tagger").labels
# for tag in dir(symbols):
#     print(f"{tag}: {spacy.explain(tag)}")
print(dir(symbols),symbols.ADV)
"""
VERB: verb, e.g. run, runs, running, eat, ate, eating
ADJ: adjective, e.g. big, old, green, incomprehensible, first
ADV: adverb, e.g. very, tomorrow, down, where, there
ADP: adposition, e.g. in, to, during
PROPN: proper noun, e.g. Mary, John, London, NATO, HBO
NOUN: noun, e.g. girl, cat, tree, air, beauty

AUX: auxiliary, e.g. is, has (done), will (do), should (do)
CONJ: conjunction, e.g. and, or, but
CCONJ: coordinating conjunction, e.g. and, or, but
DET: determiner, e.g. a, an, the
INTJ: interjection, e.g. psst, ouch, bravo, hello
NUM: numeral, e.g. 1, 2017, one, seventy-seven, IV, MMXIV
PART: particle, e.g. ’s, not,
PRON: pronoun, e.g I, you, he, she, myself, themselves, somebody
PUNCT: punctuation, e.g. ., (, ), ?
SCONJ: subordinating conjunction, e.g. if, while, that
SYM: symbol, e.g. $, %, §, ©, +, −, ×, ÷, =, :), 😝
X: other, e.g. sfpksdpsxmsa
SPACE: space, e.g.
"""
all_tags=dir(symbols)

for item in check_tag:
    is_in=False
    # pos_tags = nlp.get_pipe("tagger").labels
    for tag in dir(symbols):
        if item in tag:
            is_in=True
            print(item,'item',tag,'tag')
    if not is_in:
        print(item)
    assert is_in,'is_in'

    
    

words_anchor='a picture of small dog playing with the ball next to a yellow book in Times Square beautifully'.split()
prompt='a dog adjacent to a basket'
prompt='a picture of small dog playing with the ball next to a yellow book in Times Square beautifully'
prompt='Taewook\'s thesis'
doc = nlp(prompt)
print('\n')
for token in doc:
    pos_tag = token.pos_
    print(pos_tag,token)
print('\n')



prompt='a dog adjacent to a basket'
words_anchor=prompt.split()
check_tag=['VERB', 'ADJ','ADV','PROPN','ADP','NOUN']
for word_idx in range(len(words_anchor)):
    cap_word=words_anchor[word_idx]
    doc = nlp(cap_word)
    for token in doc:
        pos_tag = token.pos_
        if pos_tag not in check_tag:
            continue


        # ADJ: adjective (e.g., green)
        # ADV: adverb (e.g., very, tomorrow)
        # ADP: adposition (e.g., to, in, during)
        # PROPN: proper noun (e.g., Mary)
        # NOUN: noun (e.g., cat, dog)
        # if (pos_tag in ['VERB', 'ADJ','ADV','PROPN','ADP','NOUN']):
        #     break