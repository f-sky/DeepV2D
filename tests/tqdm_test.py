from tqdm import tqdm
import time

for i in tqdm(range(100)):
    for j in tqdm(range(100), leave=False,desc=''):
        time.sleep(0.1)
