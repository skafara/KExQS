from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt

# Step 1: Build Grover circuit
n = 2
qc = QuantumCircuit(n, n)

# Initialization: superposition
qc.h(range(n))

# Oracle for |11⟩: apply CZ gate (or X + CCZ + X if general)
qc.cz(0, 1)

# Diffusion operator (inversion about the mean)
qc.h(range(n))
qc.x(range(n))
qc.h(1)
qc.cx(0, 1)
qc.h(1)
qc.x(range(n))
qc.h(range(n))

# Measurement
qc.measure(range(n), range(n))

# Step 2: Simulate using Qiskit Aer
sim = AerSimulator()
compiled = transpile(qc, sim)
job = sim.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()

# Step 3: Visualize
plot_histogram(counts)
plt.title("Grover's Algorithm Output (n=2)")
plt.show()
