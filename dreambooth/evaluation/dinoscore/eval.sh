

# vase
import CUDA_VISIBLE_DEVICES=4;
python eval.py \
--fake_root='../../results/single/vase/ti_mlm0001_vase_freezemask_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/vase/"
ti_mlm0001_vase_freezemask_s3000
ti_mlm00005_vase_freezemask_masked_s3000
ti_mlm00005_vase_freezemask_s3000
ti_mlm00025_vase_freezemask_s3000
tiv2_no_mlm_vase_with_keyword_s3000


# wooden_pot
import CUDA_VISIBLE_DEVICES=4;
python eval.py \
--fake_root='../../results/single/wooden_pot/tiv2_no_mlm_wooden_pot_with_keyword_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/wooden_pot/"
ti_mlm0001_wooden_pot_freezemask_s3000
ti_mlm00005_wooden_pot_freezemask_masked_s3000
ti_mlm00005_wooden_pot_freezemask_s3000
ti_mlm00025_wooden_pot_freezemask_s3000
tiv2_no_mlm_wooden_pot_with_keyword_s3000


# backpack
import CUDA_VISIBLE_DEVICES=7;
python eval.py \
--fake_root='../../results/single/backpack/tiv2_no_mlm_backpack_with_keyword_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/backpack/"

ti_mlm0001_backpack_freezemask_s3000
ti_mlm00005_backpack_freezemask_masked_s3000
ti_mlm00005_backpack_freezemask_s3000
ti_mlm00025_backpack_freezemask_s3000
tiv2_no_mlm_backpack_with_keyword_s3000


# pet_dog1
import CUDA_VISIBLE_DEVICES=7;
python eval.py \
--fake_root='../../results/single/pet_dog1/tiv2_no_mlm_pet_dog1_with_keyword_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/pet_dog1/"
ti_mlm0001_pet_dog1_freezemask_s3000
ti_mlm00005_pet_dog1_freezemask_masked_s3000
ti_mlm00005_pet_dog1_freezemask_s3000
ti_mlm00025_pet_dog1_freezemask_s3000
tiv2_no_mlm_pet_dog1_with_keyword_s3000

# 
# pet_cat1
import CUDA_VISIBLE_DEVICES=7;
python eval.py \
--fake_root='../../results/single/pet_cat1/tiv2_no_mlm_pet_cat1_with_keyword_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/pet_cat1/"

ti_mlm0001_pet_cat1_freezemask_s3000
ti_mlm00005_pet_cat1_freezemask_masked_s3000
ti_mlm00005_pet_cat1_freezemask_s3000
ti_mlm00025_pet_cat1_freezemask_s3000
tiv2_no_mlm_pet_cat1_with_keyword_s3000

import CUDA_VISIBLE_DEVICES=7;
python eval.py \
--fake_root='../../results/single/barn/tiv2_no_mlm_barn_with_keyword_s3000/generated' \
--real_root="/data/twkim/diffusion/personalization/collected/images/barn/"
ti_mlm0001_barn_freezemask_s3000
ti_mlm00005_barn_freezemask_masked_s3000
ti_mlm00005_barn_freezemask_s3000
ti_mlm00025_barn_freezemask_s3000
tiv2_no_mlm_barn_with_keyword_s3000
