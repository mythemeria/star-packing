Example usage (Liberty Jack for a US flag):

`python starpack.py '{5/2}' 50 --background '#00205B' --border '#FFFFFF' --fill '#FFFFFF' --aspect 1.408450704225352`

Without an MPI launcher, the program runs independent search processes on the CPUs available to the current job. Use `--workers N` to set a local process count, or `--workers 1` for serial execution. `--epochs N` stops after N search epochs; the default continues until interrupted.

Under MPI, each rank searches independently and exchanges the best packing after each epoch. Local processes are disabled when MPI has multiple ranks, preventing nested oversubscription:

`mpiexec -n 64 python starpack.py '{5/2}' 50 --aspect 1.408450704225352`
