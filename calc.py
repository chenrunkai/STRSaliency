import numpy as np

while True:
    x = list(map(float, input().split(","))) 
    print("mean, std: %.2f,%.2f"%(np.mean(x), np.std(x, ddof=1)))