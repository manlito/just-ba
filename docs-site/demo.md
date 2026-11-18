# Live demo

A small incremental reconstruction, run in your browser: structure from motion on a
generated capture, from matches with outliers to adjusted poses and points. Every
image pair's matches are verified by relative pose in RANSAC; the pair with the most
inliers and enough parallax gives the first two cameras and the first points; each
further image is registered by PnP, its new points triangulated, and everything
registered is bundle-adjusted with this library. The demo's pose estimation comes from
a separate library: just-ba itself takes matches that are already verified.

The scenes have a known ground truth, so each run is checked against it: rotation
error after the best common rotation, camera and point positions after the best
similarity, as a share of the scene's size.

```{only} demo
<iframe src="demo/index.html" title="Incremental reconstruction demo" loading="lazy"
        style="width: 100%; height: 1560px; border: 0;"></iframe>

<p><a href="demo/index.html">Open the demo on its own page</a>.</p>

It needs WebGPU and WebAssembly JavaScript Promise Integration, which current Chrome
and Edge have. Without a WebGPU device it runs on the CPU only.
```

```{only} not demo
The demo is not part of this build of the documentation.
```
