# Optional sound overrides

The viewer synthesises all of its sounds in code (WebAudio): the EV motor whine, wind and tyre
noise, a two-tone horn and an original reverse chime. No audio files ship with the project.

To use your own recordings instead, drop MP3 files here, next to `index.html`:

| File          | Used for                                                                 |
|---------------|--------------------------------------------------------------------------|
| `horn.mp3`    | the horn (H, Space while driving, gamepad B); played once per press      |
| `reverse.mp3` | the reverse warning (drive mode in R, or telemetry gear R); looped       |

The viewer checks for them with a `HEAD` request when audio starts (your first click or key
press), so a missing file just means the synthesised sound is used. These files are git-ignored
(`viewer/sounds/*.mp3`), so they stay on your machine. Only use recordings you have the rights to.
