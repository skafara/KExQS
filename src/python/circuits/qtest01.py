from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# Create a quantum circuit with 3 qubits + 3 classical bits
qc = QuantumCircuit(3, 3)

# Apply gates
qc.h(0)                 # Hadamard on q0
qc.x(1)                 # Pauli-X on q1
qc.x(2)                 # Pauli-X on q2
qc.cx(1, 0)             # CNOT on q0 controlled by q1
qc.swap(0, 2)           # Swap q0 with q2
qc.t(0)                 # Pi-over-eight gate on q0
qc.p(np.pi/3.0, 1)      # Phase (pi/3) gate on q1
qc.ccx(0, 1, 2)         # Toffoli gate on q2 controlled by q0 and q1
qc.cp(np.pi/4.0, 0, 2)  # Controlled Phase (pi/4) gate on q2 controlled by q0

# Show the circuit
qc.draw('mpl')

# Get and print statevector (before measurement)
state = Statevector.from_instruction(qc)
print("Statevector:", state)

# Measure all qubits
qc.measure([0, 1, 2], [0, 1, 2])

# Simulate measurements
sim = AerSimulator(seed_simulator=42)
result = sim.run(qc, shots=1024).result()
counts = result.get_counts()

print("Measurement counts:", counts)

# Plot measurement results
plot_histogram(counts)
plt.show()
