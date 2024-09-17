import numpy as np
import time
from tqdm.auto import tqdm
progress_bar = tqdm(
        range(0, 1000),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=0,
    )
global_step=0
for i in range(1000):
    progress_bar.update(1)
    global_step+=1
    logs={'loss':np.random.rand()}
    time.sleep(0.5)
    progress_bar.set_postfix(**logs) #progress_Bar printing
    print()
