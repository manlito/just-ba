// Sizes of the shared structures as Metal lays them out, checked by the host against
// its own (backends/metal/src/backend.mm). Embedded after the shared kernels.

kernel void layout_sizes(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  out[0] = sizeof(Args);
  out[1] = sizeof(State);
  out[2] = sizeof(Op);
  out[3] = sizeof(Factor);
  out[4] = sizeof(Chunk);
  out[5] = sizeof(FBlockDesc);
  out[6] = sizeof(Tile);
  out[7] = sizeof(TileEntry);
  out[8] = sizeof(TileChunk);
  out[9] = sizeof(GatherUnit);
  out[10] = sizeof(GatherSplit);
}
