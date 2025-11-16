"""
@File    :   test.py
@Time    :   2025/11/16 09:55:12
@Author  :   Lin
@Version :   1.0
@Desc    :   None
copyright USTC
"""

import ast

with open(file="data/math/math_train.json", mode="r") as f:
    for line in f.readlines():
        # print(extract_Math(json.loads(line)["solution"])
        pass

print(ast.literal_eval("3+4"))
