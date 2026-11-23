# Real-footage run (media-real)

Machine: Mac mini (Mac16,10), Apple M4, 10 cores, 16 GB. COLMAP 4.2.1 CPU, Brush v0.3.0 on Metal. Viewer: Spark 2.2 + three.js on `main` (original white crossover, Model Y proportions), headed Google Chrome, `--use-angle=metal`, 1920x1080, frame-stepped Playwright capture (every output frame is exactly 1/30 s of viewer time). Encoded H.264 yuv420p, 30 fps, CRF 20.

Footage is the verified T7 backup of one day of TeslaCam. Recordings are of the reconstructed splat and the synthetic car only: no raw dashcam frames, mosaics, footage or .ply files are committed, and the viewer was run with `?clean=1` (no timestamp, no dashcam side panel).

## bridge

- Clip `2026-11-22_12-15-06`, window start 5 s, 20 s, 3 fps, four cameras. Daytime low-speed street (~26-28 mph), gas station, houses, bridge ahead.
- COLMAP rig model `sparse_s2m/0`: 60/60 frames, 240/240 images, 16,412 points, mean track 4.75, 0.640 px mean reprojection error.
- Brush: 30,000 steps, 4,368 s, trained without `--eval` (so no held-out PSNR).
- Files in `bridge/`: `chase.png`, `driver.png`, `free.png`, `mobile_chase_390x844.png` (1170x2532 at 3x DPR), `chase_20s.mp4`, `orbit_12s.mp4` (slow free orbit around the paused car at 0:10), `chase_driver_free_8s.mp4` (chase, driver view, free orbit that follows the car).

### Bench card (verbatim, `otrv bench bridge --note "first real run"`)

```
### bridge  (Apple M4, 16 GB)
- registration: front 60/60 · back 60/60 · left_repeater 60/60 · right_repeater 60/60
- sparse: 0.640245 px reprojection, 16412 points
- held-out PSNR: n/a (run `recon --eval`)
- time: n/a
- config: {"clip": "2026-11-22_12-15-06", "start": 5, "dur": 20, "fps": 3, "cams": 4}
- gates: PASS
```

(More scenes and the rough-edges list follow in later pushes.)
