// Voiceover lengths in seconds (from promo/voice/voice-manifest.json). Compositions size to these.
export const VOICE = {
  hero: { file: "voice/hero.mp3", seconds: 40 },
  short: { file: "voice/short.mp3", seconds: 18 },
  explainer: { file: "voice/explainer.mp3", seconds: 38 },
};

// Spoken lines, used for on-screen captions. Timed by word count across the voiceover.
export const LINES = {
  short: [
    "Your Tesla is filming everything.",
    "So I turned my dashcam into a 3D world.",
    "Four cameras. One rig.",
    "A Gaussian splat, trained on a Mac.",
    "Now I can replay my drive from any angle.",
    "It's open source.",
    "Pull your USB drive and try it.",
  ],
};

export function lineTimes(lines: string[], totalFrames: number, lead = 6) {
  const words = lines.map((l) => l.split(/\s+/).length + 1.5); // +pause weight per line
  const sum = words.reduce((a, b) => a + b, 0);
  let t = lead;
  return lines.map((text, i) => {
    const d = ((totalFrames - lead) * words[i]) / sum;
    const out = { text, from: Math.round(t), dur: Math.round(d) };
    t += d;
    return out;
  });
}
