import pickle
import numpy as np

np_array = []

with open('dataset.log', 'rb') as f:
    while True:
        try:
            data = pickle.load(f)
            step_lists = data.steps
            for i in range(len(step_lists)):
                step = step_lists[i]
                obs = step.obs
                np_array.append(obs)

        except:
            break

np_array = np.stack(np_array, axis=0)
print(np_array.shape)
np.save('dataset.npy', np_array)



print(data)