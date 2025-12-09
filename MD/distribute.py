import numpy as np

def distribute(p, N):
  ''' Distributes p indistiguishable atoms in N traps. Returns a list of the possible distributions. (A distribution is itself a list of N non-negative integers whith sum p.) '''
  if p == 0:
    return [[0,]*N]
  elif N == 1:
    return [[p]]
  else:
    return one_in_first(distribute(p-1, N)) + zero_in_first(distribute(p, N-1))

def one_in_first(l):
  ''' For all distributions in l, add one atom in the first trap. '''
  N = len(l[0])
  return [list(np.array([1] + [0]*(N-1)) + np.array(e)) for e in l]

def zero_in_first(l):
  ''' For all distributions in l, add a new empty trap. '''
  return [[0] + e for e in l]

for e in distribute(5, 5):
  print(e)