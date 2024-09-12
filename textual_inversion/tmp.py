import numpy as np
input_ids_pos=np.arange(10)
non_special_idxs = [x not in {0, 2, 4} for i, x in enumerate(input_ids_pos)]
print(non_special_idxs)