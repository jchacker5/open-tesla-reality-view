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

## street

- Clip `2026-11-22_13-22-15`, window start 20 s, 20 s, 3 fps, four cameras. Daytime two-lane street (~25-28 mph), houses tight to the sidewalk, gas station on the right.
- COLMAP rig model `sparse_c1m/0`: 60/60 frames, 240/240 images, 21,521 points, mean track 4.27, 0.866 px mean reprojection error.
- Brush: 30,000 steps with `--eval`, 6,221 s.
- Files in `street/`: `chase.png`, `driver.png`, `free.png`, `mobile_chase_390x844.png` (1170x2532 at 3x DPR), `chase_20s.mp4`, `orbit_12s.mp4` (slow free orbit around the paused car at 0:10), `chase_driver_free_8s.mp4` (chase, driver view, free orbit that follows the car, starting at 0:06).

### Bench card (verbatim, `otrv bench street --note "first real run"`)

```
### street  (Apple M4, 16 GB)
- registration: front 60/60 · back 60/60 · left_repeater 60/60 · right_repeater 60/60
- sparse: 0.866469 px reprojection, 21521 points
- held-out PSNR: 24.02 dB over 30 views (back 22.36, front 28.53, left_repeater 23.75, right_repeater 21.7)
- time: train 6221s
- config: {"clip": "2026-11-22_13-22-15", "start": 20, "dur": 20, "fps": 3, "cams": 4, "steps": 30000, "global_mapper": false, "rig": true, "eval": true}
- gates: PASS
```

### What still looks rough

Ranked by how much they break the replay:

1. Other vehicles are streaks. Parked and moving cars were not masked, so curbs and the opposing lane smear in chase and in the side views.
2. Rear and repeater views are soft and full of floaters. Right repeater held-out PSNR is 21.7 dB, back is 22.36 dB. Poles, wires, and porch edges ghost.
3. Driver view has a dark ego-hood and roof ghost across the top of the frame. The synthetic car is hidden in that mode; this is leftover splat of the real car.
4. In the free orbit the white crossover sits a little above the lane paint. The body itself is intact and stays on the road.

(Residential follows in the next push.)

