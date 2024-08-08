cp /home/twkim/anaconda3/envs/context/lib/python/site-packages/transformers/models/clip/modeling_clip.py assets/modeling_clip.py;
cp /home/twkim/anaconda3/envs/context/lib/python/site-packages/transformers/modeling_outputs.py assets/modeling_outputs.py;
cd ../;
git add .;
git commit -m "update";
git push origin main;
cd cusom_diffusion;


