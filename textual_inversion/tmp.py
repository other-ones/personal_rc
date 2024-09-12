import numpy as np
input_ids_pos=np.arange(10)
non_special_idxs = [x not in {0, 2, 4} for i, x in enumerate(input_ids_pos)]
mask_candidate_idxs=[i for i, x in enumerate(input_ids_pos) if x not in {0,2,3}]
# print(non_special_idxs)
# print(mask_candidate_idxs)
tmp=np.copy(input_ids_pos)
tmp[:5]=100
print(tmp,'tmp')
print(input_ids_pos,'input_ids_pos')
# input_ids_pos[:5]=10
out=np.random.choice([],2)
print(out)