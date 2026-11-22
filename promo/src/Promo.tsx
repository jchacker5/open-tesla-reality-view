import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { Audio } from "@remotion/media";
import { linearTiming, TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import type React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  OffthreadVideo,
  spring,
  getStaticFiles,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const { fontFamily: inter } = loadInter("normal", { weights: ["300", "500", "700"], subsets: ["latin"] });
const { fontFamily: mono } = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] });

const C = { bg: "#07080a", text: "#eef0f4", dim: "#8a909c", accent: "#3d7bff", line: "#1f232a" };
export const FPS = 30;
const T = 15; // transition frames

// Timed to the hero voiceover (voice/hero.mp3, ~31.7 s): each scene lands on its line.
export const SCENES = {
  footage: 100, // "This is a drive I took in my Tesla."
  chase: 110, // "And this is the same drive, rebuilt in 3D."
  title: 165, // "Open Tesla Reality View takes the footage..."
  pipeline: 255, // "Every camera is solved together as one rig..."
  orbit: 180, // "Ride along behind your own car. Or step out, and look around."
  cta: 240, // "It's free, it's open source... rebuild yours."
};
export const PROMO_FRAMES = Object.values(SCENES).reduce((a, b) => a + b, 0) - T * (Object.keys(SCENES).length - 1);

const ease = Easing.bezier(0.16, 1, 0.3, 1);

function useIn(delay = 0, dur = 24) {
  const f = useCurrentFrame();
  return interpolate(f - delay, [0, dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });
}

const Rise: React.FC<{ delay?: number; children: React.ReactNode; style?: React.CSSProperties }> = ({
  delay = 0,
  children,
  style,
}) => {
  const p = useIn(delay);
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 28}px)`, ...style }}>{children}</div>;
};

const Kicker: React.FC<{ children: React.ReactNode; delay?: number }> = ({ children, delay = 0 }) => (
  <Rise delay={delay}>
    <div style={{ fontFamily: mono, fontSize: 24, letterSpacing: 4, color: C.accent, textTransform: "uppercase" }}>
      {children}
    </div>
  </Rise>
);

const Caption: React.FC<{ kicker: string; title: string; sub?: string }> = ({ kicker, title, sub }) => (
  <div style={{ position: "absolute", left: 96, bottom: 88, maxWidth: 1100 }}>
    <Kicker delay={6}>{kicker}</Kicker>
    <Rise delay={10}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 72, color: C.text, lineHeight: 1.05, marginTop: 14 }}>
        {title}
      </div>
    </Rise>
    {sub && (
      <Rise delay={16}>
        <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 34, color: "#c9ced8", marginTop: 16 }}>{sub}</div>
      </Rise>
    )}
  </div>
);

const Shade: React.FC = () => (
  <AbsoluteFill
    style={{
      background:
        "linear-gradient(180deg, rgba(7,8,10,0) 45%, rgba(7,8,10,.55) 70%, rgba(7,8,10,.92) 100%), linear-gradient(90deg, rgba(7,8,10,.55), rgba(7,8,10,0) 55%)",
    }}
  />
);

const Clip: React.FC<{ src: string; zoom?: [number, number]; trim?: number }> = ({ src, zoom = [1.0, 1.06], trim = 0 }) => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const s = interpolate(f, [0, durationInFrames], zoom);
  return (
    <AbsoluteFill style={{ transform: `scale(${s})` }}>
      <OffthreadVideo src={staticFile(src)} muted trimBefore={trim} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </AbsoluteFill>
  );
};

const Title: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sp = spring({ frame: f - 4, fps, config: { damping: 200 } });
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <AbsoluteFill style={{ opacity: 0.45, filter: "blur(10px) saturate(1.2)" }}>
        <Clip src="chase.mp4" zoom={[1.15, 1.25]} trim={60} />
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(7,8,10,.35), rgba(7,8,10,.95) 75%)" }} />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center" }}>
        <Kicker delay={2}>open source · runs on your mac</Kicker>
        <div
          style={{
            fontFamily: inter,
            fontWeight: 700,
            fontSize: 132,
            letterSpacing: -4,
            color: C.text,
            marginTop: 22,
            transform: `scale(${0.92 + 0.08 * sp})`,
            opacity: sp,
          }}
        >
          Open Tesla <span style={{ color: C.accent }}>Reality View</span>
        </div>
        <Rise delay={18}>
          <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 40, color: "#c9ced8", marginTop: 18 }}>
            Your dashcam footage, rebuilt as a 3D world you can fly through.
          </div>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const Footage: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Clip src="mosaic.mp4" zoom={[1.0, 1.04]} />
    <Shade />
    <Caption kicker="dashcam · 4 cameras" title="This is a drive." sub="Straight off the USB drive in the glovebox." />
  </AbsoluteFill>
);

const STEPS = [
  { k: "ingest", t: "Copy & index", d: "rsync TeslaCam, group cameras" },
  { k: "prep", t: "Frames & masks", d: "ego-car + moving-object masks" },
  { k: "recon", t: "Camera rig SfM", d: "COLMAP, cameras solved as one rig" },
  { k: "train", t: "Gaussian splat", d: "Brush on Apple-silicon GPU" },
  { k: "view", t: "Replay", d: "chase cam, HUD, synced feeds" },
];

const Pipeline: React.FC = () => {
  const f = useCurrentFrame();
  const typed = "$ otrv recon drive1".slice(0, Math.max(0, Math.floor((f - 6) / 2)));
  return (
    <AbsoluteFill style={{ background: C.bg, padding: "0 96px", justifyContent: "center" }}>
      <Kicker>02 · reconstruct</Kicker>
      <Rise delay={4}>
        <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 72, color: C.text, marginTop: 14 }}>
          Five commands. No cloud.
        </div>
      </Rise>
      <div style={{ fontFamily: mono, fontSize: 34, color: C.dim, marginTop: 28, height: 44 }}>
        {typed}
        <span style={{ opacity: Math.floor(f / 15) % 2 ? 0 : 1, color: C.accent }}>▍</span>
      </div>
      <div style={{ display: "flex", gap: 22, marginTop: 64 }}>
        {STEPS.map((s, i) => {
          const p = interpolate(f - 30 - i * 14, [0, 22], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
            easing: ease,
          });
          const lit = f > 30 + i * 14 + 18;
          return (
            <div
              key={s.k}
              style={{
                flex: 1,
                opacity: p,
                transform: `translateY(${(1 - p) * 40}px)`,
                border: `1px solid ${lit ? C.accent : C.line}`,
                background: "#0e1014",
                borderRadius: 18,
                padding: "30px 28px",
                boxShadow: lit ? "0 0 40px rgba(61,123,255,.18)" : "none",
              }}
            >
              <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>
                {String(i + 1).padStart(2, "0")} · {s.k}
              </div>
              <div style={{ fontFamily: inter, fontWeight: 500, fontSize: 36, color: C.text, marginTop: 16 }}>{s.t}</div>
              <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 24, color: C.dim, marginTop: 10, lineHeight: 1.3 }}>
                {s.d}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const Chase: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Clip src="chase.mp4" zoom={[1.0, 1.05]} />
    <Shade />
    <Caption kicker="gaussian splat" title="This is the same drive, in 3D." sub="Rebuilt from the real camera frames, not a game engine." />
  </AbsoluteFill>
);

const Orbit: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Clip src="orbit.mp4" zoom={[1.0, 1.04]} />
    <Shade />
    <Caption kicker="explore" title="Ride along, or step out." sub="Chase cam, driver view, or fly freely through the street." />
  </AbsoluteFill>
);

const Telemetry: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Clip src="modes.mp4" zoom={[1.0, 1.03]} />
    <Shade />
    <Caption kicker="05 · telemetry" title="Speed. Steering. Autopilot." sub="Read from the metadata embedded in each clip, synced to every frame." />
  </AbsoluteFill>
);

const CTA: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center", textAlign: "center" }}>
    <AbsoluteFill style={{ opacity: 0.22, filter: "blur(14px)" }}>
      <Clip src="orbit.mp4" zoom={[1.2, 1.3]} trim={90} />
    </AbsoluteFill>
    <Kicker>MIT licensed · contributions welcome</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 96, color: C.text, marginTop: 20, letterSpacing: -2 }}>
        Rebuild your drive.
      </div>
    </Rise>
    <Rise delay={14}>
      <div
        style={{
          fontFamily: mono,
          fontSize: 38,
          color: C.text,
          marginTop: 36,
          padding: "18px 30px",
          border: `1px solid ${C.accent}`,
          borderRadius: 14,
          background: "rgba(61,123,255,.08)",
        }}
      >
        github.com/jchacker5/open-tesla-reality-view
      </div>
    </Rise>
    <Rise delay={26}>
      <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 22, color: C.dim, marginTop: 60 }}>
        Community project · not affiliated with or endorsed by Tesla, Inc.
      </div>
    </Rise>
  </AbsoluteFill>
);

export const Promo: React.FC = () => {
  const seq: [keyof typeof SCENES, React.FC][] = [
    ["footage", Footage],
    ["chase", Chase],
    ["title", Title],
    ["pipeline", Pipeline],
    ["orbit", Orbit],
    ["cta", CTA],
  ];
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      {getStaticFiles().some((f) => f.name === "voice/hero.mp3") && <Audio src={staticFile("voice/hero.mp3")} />}
      <TransitionSeries>
        {seq.flatMap(([k, Comp], i) => {
          const items = [
            <TransitionSeries.Sequence key={k} durationInFrames={SCENES[k]} premountFor={30}>
              <Comp />
            </TransitionSeries.Sequence>,
          ];
          if (i < seq.length - 1)
            items.push(
              <TransitionSeries.Transition key={`${k}-t`} presentation={fade()} timing={linearTiming({ durationInFrames: T })} />,
            );
          return items;
        })}
      </TransitionSeries>
    </AbsoluteFill>
  );
};
