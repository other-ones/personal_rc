# Vase
export CUDA_VISIBLE_DEVICES=1;
python eval_clipscore.py \
--pred_root ../../results/single/vase/tiv2_no_mlm_vase_with_keyword_s3000/generated \
--caption_path ../../results/single/vase/tiv2_no_mlm_vase_with_keyword_s3000/captions.json
tiv2_no_mlm_vase_with_keyword_s3000
ti_mlm0001_vase_unfreezemask_s3000
ti_mlm00005_vase_unfreezemask_s3000
ti_mlm00025_vase_unfreezemask_s3000

# wooden_pot 
export CUDA_VISIBLE_DEVICES=7;
python eval_clipscore.py \
--pred_root ../../results/single/wooden_pot/ti_mlm00025_wooden_pot_unfreezemask_s3000/generated \
--caption_path ../../results/single/wooden_pot/ti_mlm00025_wooden_pot_unfreezemask_s3000/captions.json

tiv2_no_mlm_wooden_pot_with_keyword_s3000
ti_mlm0001_wooden_pot_unfreezemask_s3000
ti_mlm00005_wooden_pot_unfreezemask_s3000
ti_mlm00025_wooden_pot_unfreezemask_s3000

# backpack
export CUDA_VISIBLE_DEVICES=6;
python eval_clipscore.py \
--pred_root ../../results/single/backpack/ti_mlm00025_backpack_unfreezemask_s3000/generated \
--caption_path ../../results/single/backpack/ti_mlm00025_backpack_unfreezemask_s3000/captions.json
tiv2_no_mlm_backpack_with_keyword_s3000
ti_mlm0001_backpack_unfreezemask_s3000
ti_mlm00005_backpack_unfreezemask_s3000
ti_mlm00025_backpack_unfreezemask_s3000



# pet_dog1
export CUDA_VISIBLE_DEVICES=0;
python eval_clipscore.py \
--pred_root ../../results/single/pet_dog1/ti_mlm00025_pet_dog1_unfreezemask_s3000/generated \
--caption_path ../../results/single/pet_dog1/ti_mlm00025_pet_dog1_unfreezemask_s3000/captions.json
tiv2_no_mlm_pet_dog1_with_keyword_s3000
ti_mlm0001_pet_dog1_unfreezemask_s3000
ti_mlm00005_pet_dog1_unfreezemask_s3000
ti_mlm00025_pet_dog1_unfreezemask_s3000


# pet_cat1
export CUDA_VISIBLE_DEVICES=0;
python eval_clipscore.py \
--pred_root ../../results/single/pet_cat1/ti_mlm00025_pet_cat1_unfreezemask_s3000/generated \
--caption_path ../../results/single/pet_cat1/ti_mlm00025_pet_cat1_unfreezemask_s3000/captions.json
tiv2_no_mlm_pet_cat1_with_keyword_s3000
ti_mlm0001_pet_cat1_unfreezemask_s3000
ti_mlm00005_pet_cat1_unfreezemask_masked_s3000
ti_mlm00005_pet_cat1_unfreezemask_s3000
ti_mlm00025_pet_cat1_unfreezemask_s3000





# barn
export CUDA_VISIBLE_DEVICES=0;
python eval_clipscore.py \
--pred_root ../../results/single/barn/ti_mlm00025_barn_unfreezemask_s3000/generated \
--caption_path ../../results/single/barn/ti_mlm00025_barn_unfreezemask_s3000/captions.json
tiv2_no_mlm_barn_with_keyword_s3000
ti_mlm0001_barn_unfreezemask_s3000
ti_mlm00005_barn_unfreezemask_masked_s3000
ti_mlm00005_barn_unfreezemask_s3000
ti_mlm00025_barn_unfreezemask_s3000
