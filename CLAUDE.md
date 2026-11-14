# just-ba

A minimal, high-performance bundle adjustment library in C++ with Python bindings.
The plan lives in `docs/`; start at `docs/README.md`.

## Before asking the owner anything

Read `docs/00-decisions.md`. It records every decision already made, with the
reason and what was rejected. Do not re-ask a question it answers. When a new
decision is made, add it there in the same change.

## Committing

Commit finished work by default, without asking first. This covers commits only:
do not push, and do not add or change remotes, unless asked.

## Standing rules

- CI must cost nothing: free hosted runners only. No nightly or scheduled jobs, no
  paid or GPU runners, no cloud machines started by CI. GPU tests are run by hand.
- Residuals are a closed, built-in set. Do not add a plug-in mechanism.
- No PyTorch and no intermediate framework. Keep dependencies minimal. Ceres is a
  test-only reference.
- Correctness means agreeing with Ceres Solver on the same problem.
- Feature detection, matching, and cleanup of tracker priors belong to the caller,
  not to this library.
