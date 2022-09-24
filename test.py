from pytket import Circuit
from pytket.extensions.qulacs import tk_to_qulacs
from pytket.passes import RemoveRedundancies
from qulacsvis import circuit_drawer

print(dir(Circuit))

qc = Circuit(2)
qc.H(0)
qc.H(1)
qc.CX(0, 1)
qc.Ry(1.5, 1)
qc.CX(0, 1)
qc.H(0)
qc.H(1)
qc.H(0)
qc.CX(0, 1)
qc.Ry(1.5, 0)
qc.CX(0, 1)
qc.H(0)

qc_qulacs = tk_to_qulacs(qc)

circuit_drawer(qc_qulacs, 'mpl')
