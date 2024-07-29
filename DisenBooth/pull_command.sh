cd ../;
git pull origin main;
cd DisenBooth;
cp assets/modeling_clip.py /home/twkim/anaconda3/envs/context/lib/python/site-packages/transformers/models/clip/modeling_clip.py;
cp assets/modeling_outputs.py /home/twkim/anaconda3/envs/context/lib/python/site-packages/transformers/modeling_outputs.py;