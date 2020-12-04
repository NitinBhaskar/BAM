import numpy as np

def calib_input(iter):
  data = []
  x_data = np.load('x.npy')
  x_data = x_data.reshape(-1, 32, 16, 1)
  for x in x_data:
    data.append(x)

  return {"conv2d_input": data}
