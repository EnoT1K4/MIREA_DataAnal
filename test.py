from collections import Counter
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv("/Users/dubrovskijvladislav/Downloads/МАД - Лист1-3.csv")
months = list(data["Month"].dropna().apply(lambda x : int(x)))
height = sorted(list(data["Height"].dropna().apply(lambda x : float(x.replace(',','.')))))


def calculate_groups(n):
    m = 1 + math.log2(n)
    return math.ceil(m)

# Example usage
data_points = len(height) #enter u dataset
groups = calculate_groups(data_points)
print(groups)
print(len(height))
    

def func_chunk(lst, n):
    for x in range(0, len(lst), n):
        e_c = lst[x : n + x]

        if len(e_c) < n:
            e_c = e_c + [None for y in range(n - len(e_c))]
        yield e_c

print(list(func_chunk(height, 5)))