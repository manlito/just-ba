# Command line

The repository also builds a sample command-line program, `just-ba`, which is not
part of the Python packages. It reads and writes the [problem file format](file-format.md)
and converts BAL problems:

```text
just-ba convert-bal <bal.txt> <out.jba>   convert a BAL problem to the just-ba format
just-ba info <problem>                    print sizes and the initial cost as JSON
just-ba solve <problem> [options]         solve and print a JSON summary
```

`<problem>` is a `.jba` file or a BAL `.txt` file. `just-ba` with no arguments prints
every option of `solve`: tolerances, losses, the gauge, the backend and linear
solver, threads, `--history` for the per-iteration history and `--output` for the
solved problem.
