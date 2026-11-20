# GPU cases: the problems of the Ceres parity cases (tests/ceres/CMakeLists.txt),
# each solved on the CPU and on every GPU backend built, to tight tolerances, and
# compared (gpu_check.cpp).
# Losses and gauge as in the parity case. Where the parity case does not compare
# parameters (no gauge: the end point along the free directions depends on the
# path; per-image intrinsics), neither does this one.

jba_gpu_case(simple_pinhole "--model SIMPLE_PINHOLE" "")
jba_gpu_case(pinhole "--model PINHOLE" "")
jba_gpu_case(simple_radial "--model SIMPLE_RADIAL" "")
jba_gpu_case(radial "--model RADIAL" "")
jba_gpu_case(no_gauge "--model RADIAL" "--gauge none --max-centre-diff 0 --max-rotation-diff 0")
jba_gpu_case(fixed_blocks "--model PINHOLE --fix-images 2 --fix-points 20" "")
jba_gpu_case(sigma "--model SIMPLE_RADIAL --sigma 0.7" "")
jba_gpu_case(shared_focal "--model RADIAL --free-intrinsics 1" "")
jba_gpu_case(shared_all_intrinsics "--model PINHOLE --free-intrinsics 7" "")
jba_gpu_case(per_image_intrinsics "--model RADIAL --per-image-intrinsics --free-intrinsics 5" "--max-centre-diff 0 --max-rotation-diff 0")
jba_gpu_case(intrinsics_prior "--model RADIAL --free-intrinsics 5 --intrinsics-prior 0.002,inf,0.001" "")
jba_gpu_case(intrinsics_prior_at_start "--model OPENCV --free-intrinsics 7 --intrinsics-prior 0.001,1,0.0005 --intrinsics-prior-at-start" "")
jba_gpu_case(intrinsics_prior_per_image "--model RADIAL --per-image-intrinsics --free-intrinsics 5 --intrinsics-prior 0.005,inf,0.002" "--max-centre-diff 0 --max-rotation-diff 0")
jba_gpu_case(split_intrinsics_prior "--model OPENCV --free-intrinsics 5 --images 16 --points 2500 --intrinsics-prior 0.002,inf,0.001" "")
jba_gpu_case(shared_intrinsics_split "--model SIMPLE_RADIAL --free-intrinsics 5 --images 16 --points 2500" "")
jba_gpu_case(split_fixed_blocks "--model SIMPLE_RADIAL --free-intrinsics 5 --images 16 --points 2500 --fix-images 2 --fix-points 200" "")
jba_gpu_case(split_pose_blocks "--model RADIAL --images 3 --points 20000" "")
jba_gpu_case(split_pose_blocks_free "--model RADIAL --free-intrinsics 5 --images 3 --points 20000" "")
jba_gpu_case(split_opencv "--model OPENCV --free-intrinsics 5 --images 16 --points 2500" "")
jba_gpu_case(split_per_image "--model RADIAL --per-image-intrinsics --free-intrinsics 5 --images 3 --points 20000" "--max-centre-diff 0 --max-rotation-diff 0")
jba_gpu_case(split_depth "--model RADIAL --free-intrinsics 5 --images 16 --points 2500 --depth 0.5 --depth-free 1" "")
jba_gpu_case(huber "--model SIMPLE_RADIAL --outliers 0.05" "--loss huber:2")
jba_gpu_case(cauchy "--model SIMPLE_RADIAL --outliers 0.05" "--loss cauchy:2")
jba_gpu_case(geman_mcclure "--model SIMPLE_RADIAL --outliers 0.05" "--loss geman_mcclure:2")
jba_gpu_case(opencv "--model OPENCV" "")
jba_gpu_case(opencv_fisheye "--model OPENCV_FISHEYE" "")
jba_gpu_case(opencv_free_distortion "--model OPENCV --free-intrinsics 7" "")
jba_gpu_case(fisheye_free_distortion "--model OPENCV_FISHEYE --free-intrinsics 7" "")
jba_gpu_case(relative_priors "--model PINHOLE --relative-priors 2" "")
jba_gpu_case(relative_priors_only "--model PINHOLE --points 0 --relative-priors 3" "")
jba_gpu_case(relative_priors_sigma "--model PINHOLE --relative-priors 2 --prior-sigma 0.01,0.02" "")
jba_gpu_case(relative_prior_outliers "--model SIMPLE_RADIAL --relative-priors 3 --prior-outliers 0.1" "")
jba_gpu_case(relative_prior_huber "--model SIMPLE_RADIAL --relative-priors 3 --prior-outliers 0.1" "--relative-prior-loss huber:2")
jba_gpu_case(relative_prior_no_loss "--model PINHOLE --relative-priors 2" "--relative-prior-loss none")
jba_gpu_case(global_identity "--model PINHOLE --global-priors" "")
jba_gpu_case(pose_graph "--model PINHOLE --points 0 --relative-priors 2 --global-priors --prior-outliers 0.05" "")
jba_gpu_case(global_rigid "--model PINHOLE --global-priors --global-relation rigid" "")
jba_gpu_case(global_similarity "--model PINHOLE --global-priors --global-relation similarity" "")
jba_gpu_case(relative_and_similarity "--model RADIAL --relative-priors 2 --global-priors --global-relation similarity --prior-outliers 0.05" "")
jba_gpu_case(priors_fixed_blocks "--model PINHOLE --relative-priors 2 --global-priors --fix-images 2" "")
jba_gpu_case(gravity "--model PINHOLE --gravity-priors" "")
jba_gpu_case(gravity_outliers "--model SIMPLE_RADIAL --gravity-priors --gravity-outliers 0.1" "")
jba_gpu_case(gravity_huber "--model SIMPLE_RADIAL --gravity-priors --gravity-outliers 0.1" "--gravity-loss huber:2")
jba_gpu_case(gravity_relative "--model PINHOLE --relative-priors 2 --gravity-priors --prior-outliers 0.05" "")
jba_gpu_case(gravity_similarity "--model PINHOLE --global-priors --global-relation similarity --gravity-priors" "")
jba_gpu_case(gravity_sigma "--model PINHOLE --gravity-priors --gravity-sigma 0.002" "")
jba_gpu_case(lines "--model PINHOLE --lines 40" "")
jba_gpu_case(lines_dof3 "--model PINHOLE --lines 40 --line-dof 3" "")
jba_gpu_case(lines_outliers "--model SIMPLE_RADIAL --lines 40 --line-outliers 0.1" "")
jba_gpu_case(lines_huber "--model SIMPLE_RADIAL --lines 40 --line-outliers 0.1" "--line-loss huber:2")
jba_gpu_case(lines_gravity "--model PINHOLE --lines 40 --gravity-priors --relative-priors 2" "")
jba_gpu_case(lines_dof3_opencv "--model OPENCV --lines 40 --line-dof 3 --relative-priors 2" "")
jba_gpu_case(lines_global "--model PINHOLE --lines 40 --global-priors" "")
jba_gpu_case(lines_two_groups "--model PINHOLE --points 0 --relative-priors 2 --prior-groups 2 --lines 40" "")
jba_gpu_case(lines_two_groups_dof3 "--model PINHOLE --points 0 --relative-priors 2 --prior-groups 2 --lines 40 --line-dof 3" "")
jba_gpu_case(depth "--model PINHOLE --depth 0.5" "")
jba_gpu_case(depth_range "--model PINHOLE --depth 0.5 --depth-kind range" "")
jba_gpu_case(depth_inverse "--model PINHOLE --depth 0.5 --depth-form inverse" "")
jba_gpu_case(depth_range_inverse "--model RADIAL --depth 0.5 --depth-kind range --depth-form inverse" "")
jba_gpu_case(depth_free "--model PINHOLE --depth 0.5 --depth-free 3" "")
jba_gpu_case(depth_free_scale "--model PINHOLE --depth 0.5 --depth-free 1" "")
jba_gpu_case(depth_free_offset "--model PINHOLE --depth 0.5 --depth-free 2" "")
jba_gpu_case(depth_outliers "--model SIMPLE_RADIAL --depth 0.5 --depth-outliers 0.1" "")
jba_gpu_case(depth_huber "--model SIMPLE_RADIAL --depth 0.5 --depth-outliers 0.1" "--depth-loss huber:2")
jba_gpu_case(depth_priors "--model PINHOLE --depth 0.3 --depth-free 3 --relative-priors 2 --gravity-priors" "")
jba_gpu_case(depth_fixed_blocks "--model PINHOLE --depth 0.5 --depth-free 3 --fix-images 2 --fix-points 20" "")

# Not a parity case: every image fixed and the intrinsics too, so only points are
# refined and the reduced camera system is empty (review of milestones 3 and 4,
# finding 1).
jba_gpu_case(points_only "--model PINHOLE --images 12 --fix-images 12" "")

# Not a parity case: each point seen by two neighbouring images only, so the scale
# of every link of the chain is free. The reduced camera system, formed in single
# precision, then loses definiteness on many steps, and sparse_schur must fall back
# to conjugate gradients (D92; review of milestones 3 and 4, finding 3). Costs are
# compared; parameters are not, since the end point along the free scales depends on
# the path, as in no_gauge.
jba_gpu_sparse_case(fallback "--model RADIAL --images 12 --points 300 --track-length 2"
                    "--expect-fallback --max-centre-diff 0 --max-rotation-diff 0")

# Not a parity case: BAL's shape without a download (review of milestone 5, finding
# 9): an intrinsics block per image with focal and distortion free, and a tenth of
# the points far away, so their depth is weak.
jba_gpu_case(bal_shape "--model RADIAL --images 30 --points 2000 --track-length 6 --per-image-intrinsics --free-intrinsics 5 --far-points 0.1"
             "--max-centre-diff 0 --max-rotation-diff 0")

# The fallbacks of sparse_schur (D98, D102), forced by gpu_check's test hooks on the
# pinhole case: a factorization that fails falls back to conjugate gradients for its
# step; with auto, one that runs out of memory, or a set-up that does not fit, leaves
# sparse_schur for iterative_schur; asked for explicitly, a set-up that does not fit
# is an error.
jba_gpu_sparse_check(sparse_forced_fallback pinhole "--linear-solver sparse_schur --fail-factorizations 3 --expect-fallback")
# A failure of the solver itself (not of the matrix) falls back too; three in a row
# are an error (document 15, finding 4).
jba_gpu_sparse_check(sparse_solver_failure pinhole "--linear-solver sparse_schur --fail-solver 2 --expect-fallback")
jba_gpu_sparse_check(sparse_solver_failures pinhole "--linear-solver sparse_schur --fail-solver 3 --expect-error")
jba_gpu_sparse_check(auto_sparse pinhole "--linear-solver auto --expect-solver sparse_schur")
jba_gpu_sparse_check(auto_factorization_oom pinhole "--linear-solver auto --fail-factorization-oom --expect-solver iterative_schur")
jba_gpu_sparse_check(auto_setup_oom pinhole "--linear-solver auto --fail-direct-setup --expect-solver iterative_schur")
jba_gpu_sparse_check(sparse_setup_oom pinhole "--linear-solver sparse_schur --fail-direct-setup --expect-error")
