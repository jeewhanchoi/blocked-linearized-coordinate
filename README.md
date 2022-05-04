# BLCO

BLCO is a GPU implementation of the Blocked Linearized CoOrdinate format for storing and processing sparse tensors. It performs key tensor decomposition operations along every mode (dimension) using a unified approach that works on a single tensor copy. The detailed algorithms are described in the following paper:

* **Efficient, Out-of-Memory Sparse MTTKRP on Massively Parallel Architectures**. Andy Nguyen, Ahmed E. Helal, Fabio Checconi, Jan Laukemann, Jesmin Jahan Tithi, Yongseok Soh, Teresa M. Ranadive, Fabrizio Petrini, Jeewhan Choi. In Proceedings of the ACM International Conference on Supercomputing (ICS), June 2022. [doi:10.1145/3524059.3532363](https://doi.org/10.1145/3524059.3532363). 

## Getting started
To compile the code, simply run `make` in the root directory. This will create the `cpd64` binary for running CANDECOMP/PARAFAC tensor decomposition (CPD) with a 64-bit BLCO mask. 
By default, the Intel ICPC compiler is used.

The code currently requires either Intel MKL or OpenBlas available on the system, as well as CUDA libraries. See [Settings](#settings) for more configuration options.

## Usage
You can perform CPD on a given tensor like this:
```bash
./cpd64 -k 10 --rank 16 -m 100 -i /path/to/tensor.tns
```
This runs CPD with a 64-bit ALTO mask, a rank-16 decomposition, using automatically selected GPU kernels, and a maximum number of 100 iterations (or until convergence).

For only running the matricized tensor times Khatri-Rao product (MTTKRP) use:
```bash
./cpd64 -k 10 --rank 16 -m 100 -i /path/to/tensor.tns -p -t 0 
```
This executes 100 iterations of the MTTKRP operation (`-p`) with a rank-16 decomposition on the target mode 0 (`-t 0`), i.e., the first mode.

Make sure you allocate enough huge pages if you have [activated the usage in config.mk](#transparent-huge-pages). 

A number of flags are available for GPU performance tuning:
 - `-k <kernel_id>` (1 is lvl1, 3 is lvl3, 10 is automatic selection, 12 is lvl 1 batched, 13 is atomic)
 - `-n <partitions>` (lvl3 partitions to use)
 - `--device <id>` (the device ID to use)
 - `--stream-data` (data is streamed to the GPU during MTTKRP computation)
 - `--max_block_size <elems>` (maximum block size if streaming is on)
 - `--batch` (batch lvl 1 kernels if selected)

Check out the `help` message for all possible runtime configurations.
```bash
./cpd -h
```

## Settings
All compilation settings can be changed in `config.mk`. Make sure to have no whitespace after your parameters.

#### Compiler
Currently, `ICC` and `GCC` are supported.

#### BLAS library
Currently, `MKL` or any library that conforms to the BLAS interface (tested with `OpenBlas`) is supported.

#### Transparent Huge Pages
You can activate the usage of pre-allocated THPs by setting the option `THP_PRE_ALLOCATION` to `true`. 
By default, 2M THPs are used.
To use 1G pages instead, set the `USE_1G` definition in `common.h` to `1`:
```cpp
#define USE_1G 1
```

## Contributors

#### Intel Labs

* Ahmed E. Helal (ahmed.helal@intel.com)
* Jan Laukemann  (jan.laukemann@intel.com)
* Fabio Checconi (fabio.checconi@intel.com)
* Jesmin Jahan Tithi (jesmin.jahan.tithi@intel.com)

#### University of Oregon

* Andy Nguyen (andyn@uoregon.edu)
* Yongseok Soh (ysoh@uoregon.edu)
* Jeewhan Choi (jeec@uoregon.edu)

## Licensing

BLCO is released under the MIT License. Please see the 'LICENSE' file for details.

