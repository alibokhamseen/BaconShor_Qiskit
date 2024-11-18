import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.quantumregister import Qubit
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from functools import reduce
from typing import Union

class Logical:
    """
    A class representing a logical qubit layout for a quantum register, including 
    data qubits and ancillary qubits for Z and X gauge measurements.

    Attributes:
        cols (int): Number of columns in the logical qubit grid.
        rows (int): Number of rows in the logical qubit grid.
        d_qreg (QuantumRegister): Data qubits represented as a quantum register.
        z_qreg (QuantumRegister): Ancillary qubits for Z measurements.
        x_qreg (QuantumRegister): Ancillary qubits for X measurements.
        l_qregs (list): List of all logical qubit registers [d_qreg, z_qreg, x_qreg].
        m_creg (ClassicalRegister): Classical register for measurement results.
        z_cregs (list): List to hold Z ancillary classical registers.
        x_cregs (list): List to hold X ancillary classical registers.

    Methods:
        idx_to_coord(idx: int) -> tuple:
            Converts a linear index into a coordinate (row, column) on the logical grid.

        coord_to_idx(coord: tuple) -> int:
            Converts a coordinate (row, column) into a linear index on the logical grid.

        q(col: int, row: int) -> Qubit:
            Accesses a data qubit at a specific coordinate (col, row).

        anc_z_q(col: int, row: int) -> Qubit:
            Accesses an ancillary qubit for Z stabilizers given a qubit position.

        anc_x_q(col: int, row: int) -> Qubit:
            Accesses an ancillary qubit for X stabilizers given a qubit position.
    """

    def __init__(self, dim: tuple[int] = (3, 3), name: str = "0") -> None:
        """
        Initializes a logical qubit layout.

        Args:
            dim (tuple[int]): Dimensions of the logical grid as (cols, rows). Default is (3, 3).
            name (str): Identifier for the logical qubit layout. Default is "0".
        """
        self.cols = dim[0]
        self.rows = dim[1]
        self.d_qreg = QuantumRegister(self.cols * self.rows, f"d{name}")
        self.z_qreg = QuantumRegister((self.cols - 1) * self.rows, f"Z{name}")
        self.x_qreg = QuantumRegister(self.cols * (self.rows - 1), f"X{name}")
        self.l_qregs = [self.d_qreg, self.z_qreg, self.x_qreg]
        self.m_creg = ClassicalRegister(self.cols * self.rows, f"M_{name}")
        self.z_cregs = []
        self.x_cregs = []

    def idx_to_coord(self, idx: int) -> tuple:
        """
        Converts a linear index into a coordinate (row, column) on the logical grid.

        Args:
            idx (int): Linear index of a qubit.

        Returns:
            tuple: Coordinate (row, column) corresponding to the given index.
        """
        return (idx % self.rows, idx % self.cols)

    def coord_to_idx(self, coord: tuple) -> int:
        """
        Converts a coordinate (row, column) into a linear index on the logical grid.

        Args:
            coord (tuple): Coordinate (row, column) of a qubit.

        Returns:
            int: Linear index corresponding to the given coordinate.
        """
        return coord[0] + coord[1] * self.cols

    def q(self, col, row) -> Qubit:
        """
        Accesses a data qubit at a specific coordinate (col, row).

        Args:
            col (int): Column of the desired data qubit.
            row (int): Row of the desired data qubit.

        Returns:
            Qubit: The data qubit at the specified coordinate.
        """
        return self.d_qreg[self.coord_to_idx((col, row))]

    def anc_z_q(self, col: int, row: int) -> Qubit:
        """
        Accesses an ancillary qubit for Z measurement at a specific coordinate.

        Args:
            inp (tuple): Coordinate (col, row) of the Z ancillary qubit.

        Returns:
            Qubit: The ancillary Z qubit at the specified coordinate.
        """
        cols, rows = self.cols - 1, self.rows
        idx = col * rows + row
        return self.z_qreg[idx]

    def anc_x_q(self, col: int, row: int) -> Qubit:
        """
        Accesses an ancillary qubit for X measurement at a specific coordinate.

        Args:
            inp (tuple): Coordinate (col, row) of the X ancillary qubit.

        Returns:
            Qubit: The ancillary X qubit at the specified coordinate.
        """
        cols, rows = self.cols, self.rows - 1
        idx = col + row * cols
        return self.x_qreg[idx]


class BaconShorCirc:
    """
    A class implementing the Bacon-Shor code in a quantum circuit using Qiskit.
    The class supports logical qubit initialization, syndrome extraction, 
    error correction, and simulation with noise models.

    Attributes:
        dim (tuple[int]): Dimensions of each logical qubit grid as (cols, rows).
        logicals (list[Logical]): List of logical qubits, each represented by the Logical class.
        qc (QuantumCircuit): The quantum circuit for the Bacon-Shor code.
        codespace (tuple): The logical codewords (words0, words1) defining the code space.
        lookup_z (dict): Lookup table for Z error correction based on syndromes.
        lookup_x (dict): Lookup table for X error correction based on syndromes.

    Methods:
        add_logical():
            Adds a new logical qubit to the circuit and its associated registers.

        initialize_logical(logical_idx: Union[int, str], state: str):
            Initializes a logical qubit to a specified state ('0', '1', '+', or '-').

        Z(logical_idx: Union[int, str]):
            Applies a logical Z operation on a specified logical qubit.

        X(logical_idx: Union[int, str]):
            Applies a logical X operation on a specified logical qubit.

        CX(logical1_idx: Union[int, str], logical2_idx: Union[int, str]):
            Applies a logical CX gate between two logical qubits.

        measure_logical(logical_idx: Union[int, str]):
            Measures a logical qubit into its associated classical register.

        syndrome_extraction(logical_idx: Union[int, str]):
            Performs syndrome extraction for a specified logical qubit.

        do_ec(logical_idx: Union[int, str]):
            Performs error correction (Z and X) on a specified logical qubit.

        run_with_plot(noise_model=None) -> dict:
            Simulates the quantum circuit with an optional noise model and 
            returns the raw and interpreted results, along with a histogram plot.

    Private Methods:
        _z_gauges(logical1_idx: Union[int, str]):
            Extracts Z syndrome information for error detection.

        _x_gauges(logical1_idx: Union[int, str]):
            Extracts X syndrome information for error detection.

        _z_error_correction(logical_idx: Union[int, str]):
            Performs Z error correction using the lookup table.

        _x_error_correction(logical_idx: Union[int, str]):
            Performs X error correction using the lookup table.

        _set_codespace():
            Initializes the logical codewords (code space) for error correction.

        _set_lookup():
            Creates lookup tables for error correction based on syndromes.

        _get_lookup_dict(major_axis: int) -> dict:
            Generates a dictionary mapping syndromes to correction instructions.

        _interpret(counts: dict) -> dict:
            Processes measurement results to identify logical states.

        _identify_logicals(logicals: list[str]) -> str:
            Identifies logical states based on the code space and syndromes.
    """

    def __init__(self, logical_num: int = 1, logical_dim: tuple[int] = (3, 3)) -> None:
        """
        Initializes a Bacon-Shor code implementation.

        Args:
            logical_num (int): Number of logical qubits to initialize. Default is 1.
            logical_dim (tuple[int]): Dimensions of the logical grid as (cols, rows). Default is (3, 3).
        """
        self.dim = logical_dim
        self.logicals = []
        self.qc = QuantumCircuit()
        self.codespace = None
        self.lookup_z = None
        self.lookup_x = None
        for _ in range(logical_num):
            self.add_logical()
        self._set_codespace()
        self._set_lookup()
    
    def add_logical(self) -> None:
        """
        Adds a logical qubit to the circuit and its associated registers.
        """
        name = len(self.logicals)
        l = Logical(self.dim, name)
        self.logicals.append(l)
        self.qc.add_register(*l.l_qregs)

    def initialize_logical(self, logical_idx: Union[int, str], state: str) -> None:
        """
        Initializes a logical qubit to a specified state ('0', '1', '+', or '-').

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit to initialize.
            state (str): State to initialize ('0', '1', '+', or '-').
        """
        assert state in ["0", "1", "+", "-"], f'State should be str ["0", "1", "+", "-"].'
        l = self.logicals[int(logical_idx)] 
        n, m = self.dim
        if state in ["+", "-"]:
            for row in range(m):
                self.qc.h(l.q(0, row))
                for col in range(n-1):
                    self.qc.cx(l.q(col, row), l.q(col+1, row))
            if state == "-":
                self.Z(logical_idx)
        if state in ["0", "1"]:
            if state == "1":
                self.X(logical_idx)
            for col in range(n):
                self.qc.h(l.q(col, 0))
                for row in range(m-1):
                    self.qc.cx(l.q(col, row), l.q(col, row+1))
            self.qc.h(l.d_qreg)
        self.qc.barrier(l.d_qreg)

    def Z(self, logical_idx: Union[int, str]) -> None:
        """
        Applies a logical Z operation on a specified logical qubit.

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit.
        """
        cols, rows = self.dim
        l = self.logicals[int(logical_idx)]
        for row in range(rows):
            self.qc.x(l.q(0, row))
        self.qc.barrier(l.d_qreg)

    def X(self, logical_idx: Union[int, str]) -> None:
        """
        Applies a logical X operation on a specified logical qubit.

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit.
        """
        cols, rows = self.dim
        l = self.logicals[int(logical_idx)]
        for col in range(cols):
            self.qc.x(l.q(col, 0))
        self.qc.barrier(l.d_qreg)     

    def CX(self, logical1_idx: Union[int, str], logical2_idx: Union[int, str]) -> None:
        """
        Applies a logical CX gate between two logical qubits.

        Args:
            logical1_idx (Union[int, str]): Index of the control logical qubit.
            logical2_idx (Union[int, str]): Index of the target logical qubit.
        """
        l1 = self.logicals[int(logical1_idx)]
        l2 = self.logicals[int(logical2_idx)]
        self.qc.cx(l1.d_qreg, l2.d_qreg)
        self.qc.barrier(*[l1.d_qreg, l2.d_qreg])

    def measure_logical(self, logical_idx: Union[int, str]) -> None:
        """
        Measures a logical qubit into its associated classical register.

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit.
        """
        l = self.logicals[int(logical_idx)]
        self.qc.add_register(l.m_creg)
        self.qc.measure(l.d_qreg, l.m_creg)

    def syndrome_extraction(self, logical_idx: Union[int, str]) -> None:
        """
        Performs syndrome extraction for a specified logical qubit.

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit.
        """
        self._z_gauges(logical_idx)
        self._x_gauges(logical_idx)
    
    def _z_gauges(self, logical1_idx: Union[int, str]) -> None:
        cols, rows = self.dim
        l = self.logicals[int(logical1_idx)]
        cycle = len(l.z_cregs)
        anc_creg = ClassicalRegister((cols-1)*rows, f"anc{logical1_idx}_Z_{cycle}")
        l.z_cregs.append(anc_creg)
        self.qc.add_register(anc_creg)
        for col in range(cols-1):
            for row in range(rows):
                self.qc.cx(l.q(col, row), l.anc_z_q(col, row))
        for col in range(cols-1, 0, -1):
            for row in range(rows):
                self.qc.cx(l.q(col, row), l.anc_z_q(col-1, row))
        self.qc.measure(l.z_qreg, anc_creg)    
        self.qc.reset(l.z_qreg)
        self.qc.barrier(l.d_qreg)
    
    def _x_gauges(self, logical1_idx: Union[int, str]) -> None:
        cols, rows = self.dim
        l = self.logicals[int(logical1_idx)]
        cycle = len(l.x_cregs)
        anc_creg = ClassicalRegister(cols*(rows-1), f"anc{logical1_idx}_X_{cycle}")
        l.x_cregs.append(anc_creg)
        self.qc.add_register(anc_creg)
        self.qc.h(l.x_qreg)
        for row in range(rows-1):
            for col in range(cols):
                self.qc.cx(l.anc_x_q(col, row), l.q(col, row))
        for row in range(rows-1, 0, -1):
            for col in range(cols):
                self.qc.cx(l.anc_x_q(col, row-1), l.q(col, row))
        self.qc.h(l.x_qreg)
        self.qc.measure(l.x_qreg, anc_creg)
        self.qc.reset(l.x_qreg)
        self.qc.barrier(l.d_qreg)

    def do_ec(self, logical_idx: Union[int, str]) -> None:
        """
        Performs error correction (Z and X) on a specified logical qubit.

        Args:
            logical_idx (Union[int, str]): Index of the logical qubit.
        """
        self._z_error_correction(logical_idx)
        self._x_error_correction(logical_idx)
    
    def _z_error_correction(self, logical_idx: Union[int, str]) -> None:
        n, m = self.dim
        st_num = n - 1
        st_len = m
        creg_len = st_num * st_len
        l = self.logicals[int(logical_idx)]
        creg = l.z_cregs[-1]
        with self.qc.switch(creg) as case:
            for gauges_int in range(2**creg_len):
                gauges_bin = f"{gauges_int:0{creg_len}b}"
                sts = ''.join(str(sum(map(int, gauges_bin[i:i + m])) % 2) for i in range(0, len(gauges_bin), m))
                sts = int(sts, 2)
                if sts in self.lookup_z.keys():
                    with case(gauges_int):
                        for col in self.lookup_z[sts]:
                            self.qc.x(l.q(col, 0))
        self.qc.barrier(l.d_qreg)
    
    def _x_error_correction(self, logical_idx: Union[int, str]) -> None:
        m, n = self.dim
        st_num = n - 1
        st_len = m
        creg_len = st_num * st_len
        l = self.logicals[int(logical_idx)]
        creg = l.x_cregs[-1]
        with self.qc.switch(creg) as case:
            for gauges_int in range(2**creg_len):
                gauges_bin = f"{gauges_int:0{creg_len}b}"
                sts = ''.join(str(reduce(lambda x, y: x ^ y, map(int, gauges_bin[i:i + m]))) for i in range(0, len(gauges_bin), m))
                sts = int(sts, 2)
                if sts in self.lookup_x.keys():
                    with case(gauges_int):
                        for row in self.lookup_z[sts]:
                            self.qc.z(l.q(0, row))
        self.qc.barrier(l.d_qreg)
    
    def _set_codespace(self) -> None:
        if self.codespace != None:
            return None
        n, m = self.dim
        backbone_0_col = [f'{i:0{m}b}' for i in range(2**m) if bin(i).count("1") % 2 == 0]
        backbone_1_col = [f'{i:0{m}b}' for i in range(2**m) if bin(i).count("1") % 2 == 1]
        words0 = [''.join(''.join(row) for row in zip(*combo)) 
                for combo in np.array(np.meshgrid(*[backbone_0_col]*n)).T.reshape(-1, n)]
        words1 = [''.join(''.join(row) for row in zip(*combo))
                for combo in np.array(np.meshgrid(*[backbone_1_col]*n)).T.reshape(-1, n)]
        self.codespace = (words0, words1)
    
    def _set_lookup(self) -> None:
        n, m = self.dim
        self.lookup_z = self._get_lookup_dict(n)
        self.lookup_x = self._get_lookup_dict(m)
    
    def _get_lookup_dict(self, major_axis) -> dict:
        max_e = (major_axis - 1) // 2
        all_combinations = np.array(np.meshgrid(*[[0, 1]] * major_axis)).T.reshape(-1, major_axis)
        valid_combinations = all_combinations[np.sum(all_combinations, axis=1) <= max_e]
        lookup_dict = {
            int(''.join(map(str, np.bitwise_xor(row[:-1], row[1:]))), 2): [len(row) - 1 - i for i, bit in enumerate(row) if bit]
            for row in valid_combinations
            }
        return lookup_dict

    def run_with_plot(self, noise_model=None) -> dict:
        """
        Simulates the quantum circuit with an optional noise model.

        Args:
            noise_model: A noise model to apply during simulation.

        Returns:
            dict: Raw counts, interpreted results, and a histogram plot.
        """
        raw_counts = AerSimulator(noise_model=noise_model, shots=2000, method="stabilizer").run(self.qc).result().get_counts()        
        results = self._interpret(raw_counts)
        return raw_counts, results, plot_histogram(results)

    def _interpret(self, counts: dict) -> dict:
        n ,m = self.dim
        l_num = len(self.logicals)
        l_len = n * m
        results = {}
        for key, cts in counts.items():
            key = key.replace(" ", "")
            key = key[:l_num * l_len]
            raw_logicals = [key[i:i+l_len] for i in range(0, l_len*l_num, l_len)][-1::-1]
            logicals = self._identify_logicals(raw_logicals)
            if logicals in results.keys():
                results[logicals] += cts
            else:
                results[logicals] = cts
        return results
        
    def _identify_logicals(self, logicals: list[str]) -> str:
        n, m = self.dim
        max_z_e = (n - 1) // 2
        results = []
        for logical in logicals:
            if logical in self.codespace[1] or ( any(sum(1. for a, b in zip(logical, s) if a != b) <= max_z_e for s in self.codespace[1])):
                results.append("1")
            elif logical in self.codespace[0] or (any(sum(1 for a, b in zip(logical, s) if a != b) <= max_z_e for s in self.codespace[0])):
                results += "0"
            else:
                results.append("F")
        return "".join(results[-1::-1])
        
