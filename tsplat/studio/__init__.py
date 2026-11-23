"""`otrv studio`: a local-first web app over the otrv pipeline.

Pick TeslaCam footage (indexed in place), see drives, choose a window, build a Gaussian-splat city with
live progress and open it in the viewer. The server binds 127.0.0.1 only and nothing leaves the machine.

progress.py and jobs.py need only the standard library; server.py needs the `studio` extra (FastAPI).
"""
