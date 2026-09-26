# Star packing

I made this for fun on a Saturday and put it online in case someone else wants to work on packing stars into rectangles. The repository includes example packings and a program that searches for smaller enclosing rectangles.

This is an incomplete experiment. The included packings are valid numerical examples, not proven optimal solutions. From what I've seen, the program sometimes misses better arrangements that are fairly easy to spot, so I expect many of these packings can be improved. A few look close to especially good arrangements; it may be possible to describe their star-to-star and star-to-edge contacts precisely and turn them into exact constructions. I haven't done that or established lower bounds.

If this problem interests you, the code and example solutions are here to use and improve. Better packings, exact constructions, proofs, and improvements to the search are all welcome.

Example usage (Liberty Jack for a US flag):

`python starpack.py '{5/2}' 50 --background '#00205B' --border '#FFFFFF' --fill '#FFFFFF' --aspect 1.408450704225352`

To render the packing on a complete US flag, run `python starpack.py '{5/2}' 50 --us-flag`. This uses a 1.9:1 flag with 13 alternating red and white stripes and a blue canton 7 stripes tall and 0.76 flag heights wide. The canton packing aspect is 0.76 × 13 / 7; `--us-flag` overrides `--aspect`, `--background`, `--border` and `--fill`. Results are saved as `results/5-2_50_us_flag.json` and `results/5-2_50_us_flag.png`. When no flag result exists, an existing `results/5-2_50.json` packing is fitted to the canton and used as a starting point.

Without an MPI launcher, the program runs independent search processes on the CPUs available to the current job. Use `--workers N` to set a local process count, or `--workers 1` for serial execution. `--epochs N` stops after N search epochs; the default continues until interrupted.

When no result file exists, the program saves a valid regular tiling before searching. For 50 `{5/2}` stars it starts from the staggered, alternating five-column pattern and fits it to the requested aspect; other symbols start from a staggered triangular lattice. An existing result is loaded instead. `--start-height` can explicitly set the initial search height without changing the saved tiling.

Every ten epochs, the current best packing attempts to set stars within 3 degrees of upright to an exact upright orientation. Only valid arrangements that fit within the current box are saved. Configure this with `--straighten-every N` and `--straighten-angle DEGREES`, or set `--straighten-every 0` to disable it.

Under MPI, each rank searches independently and exchanges the best packing after each epoch. Local processes are disabled when MPI has multiple ranks, preventing nested oversubscription:

`mpiexec -n 64 python starpack.py '{5/2}' 50 --aspect 1.408450704225352`
