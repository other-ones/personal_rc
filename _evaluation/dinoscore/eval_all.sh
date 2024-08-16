export CUDA_VISIBLE_DEVICES=7;
python eval_all.py \
--method_root='./../../custom_diffusion' \
--keywords='single/-_resume_s' \
--grounded=1


export CUDA_VISIBLE_DEVICES=7;
python eval_all.py \
--method_root='./../../textual_inversion' \
--keywords='single_prior' \
--grounded=1 \
--sort=0