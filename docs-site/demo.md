# Live demo

Structure from motion on real photos, in your browser. The demo finds keypoints (ALIKED or
XFeat, through just-keypoints) and matches every pair of images (LightGlue, or XFeat's
LighterGlue, through just-matching) on your GPU. Then it reconstructs them incrementally: each pair's matches
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
and Edge have. The first run downloads LightGlue's weights (24 MB), or LighterGlue's (4.5 MB)
with XFeat. Your own photos stay
in the browser: nothing is uploaded.
```

```{only} not demo
The demo is not part of this build of the documentation.
```

## Live capture

The same pipeline on your camera, as you capture: open the page on a laptop or a phone,
point the camera at a scene and walk around it. The page keeps a keyframe each time the
view has moved on enough. It uses Like Recon's rules: how much of the last keyframe is
still in view, how many verified matches link the two, the keypoints and the blur. Each
keyframe goes to a map that grows while you move (the demo's mapper session: verified
pairs, placement by PnP, a window of the latest keyframes bundle-adjusted with this
library). Finish then matches more pairs and reconstructs every keyframe, starting from
that map.

Matches come from FlexGlue, LightGlue or LighterGlue (with XFeat keypoints instead of
ALIKED's). FlexGlue gives the better map. Its weights are not
served by this site, because the backbone is under Meta's DINOv3 License. You download
`flexglue.pth` (567 MB) from FlexGlue's release once and give it to the page, which
checks it, converts it to half precision (284 MB) and keeps it in the browser. LightGlue's
weights (24 MB) come with the page, and so do LighterGlue's (4.5 MB), the default on phones,
where FlexGlue is heavy: XFeat and LighterGlue are the lightest pair.

```{only} demo
<p><a href="demo/capture.html"><strong>Open the live capture</strong></a>. On a computer,
you can also open this page on your phone.</p>

It needs WebGPU, WebAssembly JavaScript Promise Integration and camera access: Chrome and
Edge 137 or later, on a computer or on Android. Safari has WebGPU but not JSPI yet, so
the page says it can't run there. The camera's frames stay in the browser: nothing is
uploaded.
```
