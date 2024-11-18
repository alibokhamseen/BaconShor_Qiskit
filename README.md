# BaconShor_Qiskit
The Bacon-Shor Code is a quantum error correction scheme that provides a simple yet effective method for detecting and correcting errors in quantum systems. This repository implements the Bacon-Shor code using Qiskit, designed to be simple but not efficient due to Qiskit simulators' nature.

## Error Correction Decoder
My implementation corrects errors in a manner similar to repetition code. I force stabilizers to be 0s after extracting syndromes. This is always possible except for even distances when the number of errors is half the distance. In situations like this, it's 50% 50% chance. So, I simply do nothing.


## Features

- Initialization of logical qubits in computational and superposition states.
- Syndrome extraction using Z and X gauge measurements.
- Error correction with precomputed lookup tables.
- Support for logical gates (`Z`, `X`, and `CX`) on logical qubits.
- Simulation with noise models and automated result interpretation.


License
This project is licensed under the MIT License. See the LICENSE file for details.
