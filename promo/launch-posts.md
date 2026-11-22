# Launch posts for X (drafts, not posted)

Pair each post with the matching render from `promo/out/`. X supports up to 2:20 per video;
1080x1920 shorts and 1920x1080 / 1080x1080 videos all play inline.

## 1. Main launch thread (video: `short.mp4`, vertical)

**Post 1**
My Tesla films everything on a USB stick in the glovebox.

So I turned the dashcam into a 3D world I can fly through. Same drive, any angle.

Open source, runs on a Mac. 🧵

**Post 2** (video: `loop.mp4`)
Top: the raw dashcam. Bottom: the same moment, rebuilt as a Gaussian splat.

All four cameras are solved together as one rigid rig, then the splat trains on the Mac's GPU. Nothing goes to the cloud.

**Post 3** (video: `explainer.mp4`)
How it works, in four commands:

otrv ingest → copy TeslaCam off the drive
otrv prep → frames + masks (your car's body and moving traffic removed)
otrv recon → COLMAP rig solve + Brush splat training
otrv view → chase cam, raw feeds and a telemetry HUD

**Post 4**
It reads the speed, steering and Autopilot telemetry Tesla embeds in every clip (format published at github.com/teslamotors/dashcam).

Works on 4-camera (HW3) and 6-camera (HW4) cars.

**Post 5**
We're building it in public. Help wanted on moving-car splats, GPS-anchored reconstruction, Linux and Windows support, and stitching long drives.

github.com/jchacker5/open-tesla-reality-view

Not affiliated with Tesla.

## 2. Standalone hooks (one video each, for the following days)

- "POV: you can rewatch your commute from outside the car." + `orbit` clip
- "Your Tesla's 4 cameras → one 3D scene. Here's the raw footage vs the reconstruction." + `loop.mp4`
- "Gaussian splatting, but the camera rig is a Model 3." + `promo.mp4`
- "I asked my car's dashcam to become a video game level." + chase clip

## Posting notes

- Lead with the video and keep the first line under ~80 characters.
- Put the repo link in a reply or the last post of the thread, since X tends to show posts with links in the first post to fewer people.
- Tag tools we use when relevant: COLMAP, Brush, Spark/World Labs. Don't tag Tesla accounts in a way that implies endorsement.
- Before posting, check the footage for anything identifying: plates, faces, the home location.
