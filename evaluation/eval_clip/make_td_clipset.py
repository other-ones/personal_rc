import os
import shutil
import numpy as np

# 'TMDBEval500'
# 'LAIONEval4000'
# 'DrawTextCreative'
# 'ChineseDrawText'
# 'OpenLibraryEval500'
# 'DrawBenchText'

td_root='/data/dataset/ocr/examples/textdiffuser/'
subsets=[
    'TMDBEval500',
    'LAIONEval4000',
    'DrawTextCreative',
    'ChineseDrawText',
    'OpenLibraryEval500',
    'DrawBenchText',
    ]
dst_root='/data/dataset/ocr/examples/textdiffuser/samples'
os.makedirs(dst_root,exist_ok=True)
for sb in subsets:
    sbpath=os.path.join(td_root,sb)
    seed=np.random.choice([0,1,2,3])
    seed_path=os.path.join(sbpath,'images_{}'.format(seed))
    flist=os.listdir(seed_path)
    for f in flist:
        src_path=os.path.join(seed_path,f)
        srcname=f.split('_')[0]
        dstname='{}_{}.png'.format(sb,srcname)
        dstpath=os.path.join(dst_root,dstname)
        os.symlink(src_path,dstpath)