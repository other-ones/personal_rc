path1='pet/cat_breeds.txt'
path2='pet/cat.txt'
breeds=[item.strip() for item in open(path1).readlines()]
captions=[item.strip() for item in open(path2).readlines()]
dst_file=open('pet/cat_prior.txt','w')
new_list=[]
for breed in breeds:
    for caption in captions:
        caption=caption.strip()
        new_caption=caption.replace('<new1>',breed)
        new_list.append(new_caption)
for line in new_list:
    dst_file.write('{}\n'.format(line))
