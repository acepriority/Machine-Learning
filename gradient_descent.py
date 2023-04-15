import numpy as np

x = np.random.randn(10, 1)
y = 2*x + np.random.rand()

w = 0.0
b = 0.0
lr = 0.01

def descent(x, y, w, b, lr):
  dldw = 0.0
  dldb = 0.0
  n = x.shape[0]

  for xi, yi in zip(x, y):
    dldw += -2*xi*(yi - (w*xi + b))
    dldb += -2*(yi - (w*xi + b))

  w = w - lr*(1/n)*dldw
  b = b - lr*(1/n)*dldb

  return w, b

for epoch in range(400):
  w, b = descent(x, y, w, b, lr)
  z = w*x + b
  loss = np.divide(np.sum((y - z)**2, axis=0), x.shape[0])
  print(f'{epoch} loss {loss}, parameters w: {w} and b: {b}')