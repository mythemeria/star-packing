Example usage (Liberty Jack for a US flag):

`python starpack.py '{5/2}' 50 --background '#00205B' --border '#FFFFFF' --fill '#FFFFFF' --aspect 1.408450704225352`

To render the packing on a complete US flag, run `python starpack.py '{5/2}' 50 --us-flag`. This uses a 1.9:1 flag with 13 alternating red and white stripes and a blue canton 7 stripes tall and 0.76 flag heights wide. The canton packing aspect is 0.76 × 13 / 7; `--us-flag` overrides `--aspect`, `--background`, `--border` and `--fill`. Results are saved as `results/5-2_50_us_flag.json` and `results/5-2_50_us_flag.png`. When no flag result exists, an existing `results/5-2_50.json` packing is fitted to the canton and used as a starting point.

Without an MPI launcher, the program runs independent search processes on the CPUs available to the current job. Use `--workers N` to set a local process count, or `--workers 1` for serial execution. `--epochs N` stops after N search epochs; the default continues until interrupted.

When no result file exists, the program saves a valid regular tiling before searching. For 50 `{5/2}` stars it starts from the staggered, alternating five-column pattern and fits it to the requested aspect; other symbols start from a staggered triangular lattice. An existing result is loaded instead. `--start-height` can explicitly set the initial search height without changing the saved tiling.

Every ten epochs, the current best packing attempts to set stars within 3 degrees of upright to an exact upright orientation. Only valid arrangements that fit within the current box are saved. Configure this with `--straighten-every N` and `--straighten-angle DEGREES`, or set `--straighten-every 0` to disable it.

Under MPI, each rank searches independently and exchanges the best packing after each epoch. Local processes are disabled when MPI has multiple ranks, preventing nested oversubscription:

`mpiexec -n 64 python starpack.py '{5/2}' 50 --aspect 1.408450704225352`
