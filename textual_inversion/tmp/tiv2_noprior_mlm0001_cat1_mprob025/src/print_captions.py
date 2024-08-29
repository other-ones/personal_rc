concept='teddybear'
path='/home/twkim/project/custom-diffusion/customconcept101/prompts/{}.txt'.format(concept)
lines=open(path).readlines()
print("[")
for line in lines:
    line=line.strip()
    line=line.replace('<new1> {}'.format(concept),'{}')
    print("\"{}\",".format(line))
print("]")
    