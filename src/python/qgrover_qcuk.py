print('\nGrovers Algorithm')
print('------------------\n')

import math
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

pi = math.pi
q = QuantumRegister(4,'q')
c = ClassicalRegister(4,'c')
qc = QuantumCircuit(q,c)

print('\nInitialising Circuit...\n')

### Initialisation ###

qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])

print('\nPreparing Oracle circuit....\n')

### 0000 Oracle ###

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

'''
### 0001 Oracle ###

qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
'''

'''
### 0010 Oracle ###

qc.x(q[0])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[2])
qc.x(q[3])
'''

'''
### 0011 Oracle ###

qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[2])
qc.x(q[3])
'''

'''
### 0100 Oracle ###

qc.x(q[0])
qc.x(q[1])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
qc.x(q[3])
'''

'''
### 0101 Oracle ###

qc.x(q[1])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[1])
qc.x(q[3])
'''

'''
### 0110 Oracle ###

qc.x(q[0])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[3])
'''

'''
### 0111 Oracle ###

qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[3])
'''

'''
### 1000 Oracle ###

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
'''

'''
### 1001 Oracle ###

qc.x(q[1])
qc.x(q[2])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[1])
qc.x(q[2])
'''

'''
### 1010 Oracle ###

qc.x(q[0])
qc.x(q[2])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[2])
'''

'''
### 1011 Oracle ###

qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[3])
'''

'''
### 1100 Oracle ###

qc.x(q[0])
qc.x(q[1])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
'''

'''
### 1101 Oracle ###

qc.x(q[1])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[1])
'''

'''
### 1110 Oracle ###

qc.x(q[0])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
'''

'''
###1111 Oracle###

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
'''

print('\nPreparing Amplification circuit....\n')
#### Amplification ####

qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])
qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])

qc.cp(pi/4, q[0], q[3])
qc.cx(q[0], q[1])
qc.cp(-pi/4, q[1], q[3])
qc.cx(q[0], q[1])
qc.cp(pi/4, q[1], q[3])
qc.cx(q[1], q[2])
qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])
qc.cx(q[1], q[2])

qc.cp(-pi/4, q[2], q[3])
qc.cx(q[0], q[2])
qc.cp(pi/4, q[2], q[3])

qc.x(q[0])
qc.x(q[1])
qc.x(q[2])
qc.x(q[3])
qc.h(q[0])
qc.h(q[1])
qc.h(q[2])
qc.h(q[3])

# Show the circuit
qc.draw('mpl')

### Measurment ###
qc.barrier(q)
qc.measure(q[0], c[0])
qc.measure(q[1], c[1])
qc.measure(q[2], c[2])
qc.measure(q[3], c[3])

### Simulation ###
print('\nExecuting job....\n')

sim = AerSimulator()
compiled = transpile(qc, sim)
job = sim.run(compiled, shots=10000)
counts = job.result().get_counts()

print('RESULT: ',counts,'\n')
plot_histogram(counts)
plt.show()
