import os
root='/home/twkim/project/rich_context/datasets_pkgs/captions/v7'
types=os.listdir(root)
for type in types:
    type_path=os.path.join(root,type)
    flist=os.listdir(type_path)
    num=0
    for ff in flist:
        fpath=os.path.join(type_path,ff)
        lines=open(fpath).readlines()
        num+=len(lines)
        print('{}\t{}'.format(ff,len(lines)))
    print('{}\t{}'.format(type,num))