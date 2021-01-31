from tqdm import tqdm
import time

for i in tqdm(100):
    for j in tqdm(100, leave=False):
        time.sleep(0.1)
