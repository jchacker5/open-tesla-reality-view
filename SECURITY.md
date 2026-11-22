# Security policy

Open Tesla Reality View runs locally. It reads dashcam files, runs ffmpeg, COLMAP and Brush, and
serves the viewer on `127.0.0.1` only.

## Reporting a vulnerability

Please **do not** open a public issue. Report it privately through GitHub:
**Security → Report a vulnerability** on this repository (a private security advisory).
Include the affected version or commit, steps to reproduce, and the impact.
We aim to acknowledge reports within a week.

Only the latest `main` is supported.

If you report a problem that involves dashcam footage, don't attach real footage. It contains GPS
traces, licence plates and faces.
