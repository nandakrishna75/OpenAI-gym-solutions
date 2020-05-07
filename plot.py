import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import re

data = []
record_filename = './record.dat'
display_data = False
if len(sys.argv) == 2:
    record_filename = sys.argv[1]
elif len(sys.argv) == 3:
    record_filename = sys.argv[1]
    display_data = int(sys.argv[2])
elif len(sys.argv) > 3:
    print("Too many arguments passed")
    exit(0)

with open(record_filename, 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass

data = np.array(data)
if data.shape[1] == 3:
    data = pd.DataFrame(data,columns=['Runs','Score','Expl'])
else:
    data = pd.DataFrame(data,columns=['Runs','Score'])

if display_data:
    print(data)
data.plot(x='Runs', y='Score', style='-o')
plt.show()

# data = []
# with open('record_ddpg_800.txt','r') as f:
#     try:
#         for i in range(825):
#             line = f.readline()
#             # line2 = f.readline()
#             val = re.split(r',| |\n',line)
#             print(val)
#             data.append([int(val[1]),float(val[4])])
#     except EOFError:
#         pass

# with open('record599999.dat','ab') as f:
#     for i in range(825):
#         pickle.dump(data[i],f)
