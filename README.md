# DeterministicAtomicAdd_CUDA
To achieve deterministic floating point addtion results, one possible solution might be performing a conversion before and after the add operation. My implementation is converting double to unsigned long long integer on each GPU processor, and then performing atomic addtion. Launching another kernel to sum them up. 
