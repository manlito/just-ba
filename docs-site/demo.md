# Live demo

Structure from motion on real photos, in your browser. The demo finds keypoints (ALIKED,
through just-keypoints) and matches every pair of images (LightGlue, through
just-matching) on your GPU. Then it reconstructs them incrementally: each pair's matches
are verified by relative pose in RANSAC, the pair with the most inliers and enough parallax
gives the first two cameras and the first points, and each further image is registered by
PnP, its new points triangulated, and everything registered is bundle-adjusted with this
library. Focal length and radial distortion are refined as it goes.

The reconstruction runs on the CPU (one thread of WebAssembly) or on WebGPU, by default
once on each with the same matches, and the page shows both times side by side. On the 44
sample images, ETH3D's *delivery_area*, the GPU's advantage is plain: verifying the 855
matched pairs alone takes the CPU tens of seconds.

The demo's pose estimation comes from a separate library, poselib-gpu; keypoints and
matches come from two more. just-ba itself takes matches that are already verified.

```{only} demo
<p><a href="demo/index.html"><strong>Open the live demo</strong></a> (also linked at the top right of every page).</p>

It needs WebGPU and WebAssembly JavaScript Promise Integration, which current Chrome
and Edge have. The first run downloads LightGlue's weights (24 MB). Your own photos stay
in the browser: nothing is uploaded.
```

```{only} not demo
The demo is not part of this build of the documentation.
```
