// "otrv render" launch clip: 1920x1080, ~26 s. The footage comes from `otrv render` itself (git-ignored,
// regenerate it for your own run <name>):
//   otrv render <name> --start 0 --end 10 --keep-frames --out promo/public/render/raw.mp4
//   otrv render <name> --start 0 --end 10 --enhance difix --accept-nvidia-license --out promo/public/render/difix.mp4
// raw.mp4 and difix.mp4 must cover the same window: Compare plays them frame-synced.
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { Video } from "@remotion/media";
import { linearTiming, TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import type React from "react";
import { AbsoluteFill, Easing, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";

export const { fontFamily: inter } = loadInter("normal", { weights: ["300", "500", "700"], subsets: ["latin"] });
export const { fontFamily: mono } = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] });

export const C = { bg: "#07080a", text: "#eef0f4", dim: "#8a909c", accent: "#3d7bff", ai: "#3ecf8e", line: "#1f232a" };
export const RENDER_FPS = 30;
const T = 15;

const SCENES = { title: 90, command: 105, compare: 255, chase: 180, cta: 120 };
export const RENDER_FRAMES =
  Object.values(SCENES).reduce((a, b) => a + b, 0) - T * (Object.keys(SCENES).length - 1);

export const ease = Easing.bezier(0.16, 1, 0.3, 1);
export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const useIn = (delay = 0, dur = 24) => interpolate(useCurrentFrame() - delay, [0, dur], [0, 1], { ...clamp, easing: ease });

export const Rise: React.FC<{ delay?: number; children: React.ReactNode; style?: React.CSSProperties }> = ({ delay = 0, children, style }) => {
  const p = useIn(delay);
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 28}px)`, ...style }}>{children}</div>;
};

export const Kicker: React.FC<{ children: React.ReactNode; delay?: number; color?: string }> = ({ children, delay = 0, color = C.accent }) => (
  <Rise delay={delay}>
    <div style={{ fontFamily: mono, fontSize: 24, letterSpacing: 4, color, textTransform: "uppercase" }}>{children}</div>
  </Rise>
);

const Caption: React.FC<{ kicker: string; title: string; sub?: string }> = ({ kicker, title, sub }) => (
  <div style={{ position: "absolute", left: 96, bottom: 88, maxWidth: 1200 }}>
    <Kicker delay={6}>{kicker}</Kicker>
    <Rise delay={10}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 68, color: C.text, lineHeight: 1.05, marginTop: 14 }}>{title}</div>
    </Rise>
    {sub && (
      <Rise delay={16}>
        <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 32, color: "#c9ced8", marginTop: 16 }}>{sub}</div>
      </Rise>
    )}
  </div>
);

export const Shade: React.FC = () => (
  <AbsoluteFill
    style={{
      background:
        "linear-gradient(180deg, rgba(7,8,10,0) 50%, rgba(7,8,10,.6) 75%, rgba(7,8,10,.92) 100%), linear-gradient(90deg, rgba(7,8,10,.5), rgba(7,8,10,0) 50%)",
    }}
  />
);

export const Clip: React.FC<{ src: string; zoom?: [number, number]; trim?: number }> = ({ src, zoom = [1.0, 1.05], trim = 0 }) => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  return (
    <AbsoluteFill style={{ transform: `scale(${interpolate(f, [0, durationInFrames], zoom)})` }}>
      <Video src={staticFile(src)} muted trimBefore={trim} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </AbsoluteFill>
  );
};

const Title: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sp = spring({ frame: f - 4, fps, config: { damping: 200 } });
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <AbsoluteFill style={{ opacity: 0.4, filter: "blur(12px) saturate(1.2)" }}>
        <Clip src="render/difix.mp4" zoom={[1.15, 1.22]} />
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(7,8,10,.3), rgba(7,8,10,.95) 75%)" }} />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center" }}>
        <Kicker delay={2}>new in open tesla reality view</Kicker>
        <div style={{ fontFamily: mono, fontWeight: 500, fontSize: 140, letterSpacing: -4, color: C.text, marginTop: 22, transform: `scale(${0.92 + 0.08 * sp})`, opacity: sp }}>
          otrv <span style={{ color: C.accent }}>render</span>
        </div>
        <Rise delay={16}>
          <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 40, color: "#c9ced8", marginTop: 18 }}>
            Your dashcam drive, rebuilt in 3D and rendered like a film.
          </div>
        </Rise>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

const CMD = "$ otrv render drive1 --enhance difix";
const Command: React.FC = () => {
  const f = useCurrentFrame();
  const typed = CMD.slice(0, Math.max(0, Math.floor((f - 6) / 1.4)));
  const done = typed.length === CMD.length;
  const lines = ["capturing 300 frames in headless Chromium (Metal)", "Difix clean-up on the Apple GPU (MPS, fp16)", "flow-guided anti-flicker", "encoding H.264 1080p"];
  return (
    <AbsoluteFill style={{ background: C.bg, padding: "0 120px", justifyContent: "center" }}>
      <Kicker>one command</Kicker>
      <div style={{ fontFamily: mono, fontSize: 52, color: C.text, marginTop: 30, height: 70 }}>
        {typed}
        <span style={{ opacity: Math.floor(f / 15) % 2 ? 0 : 1, color: C.accent }}>▍</span>
      </div>
      <div style={{ marginTop: 40 }}>
        {lines.map((l, i) => {
          const p = interpolate(f - 62 - i * 9, [0, 14], [0, 1], { ...clamp, easing: ease });
          return (
            <div key={l} style={{ fontFamily: mono, fontSize: 30, color: C.dim, opacity: done ? p : 0, marginTop: 12 }}>
              <span style={{ color: C.ai }}>✓</span> {l}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// Raw splat vs Difix, frame-synced, with a moving divider.
const Compare: React.FC = () => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const x = interpolate(f, [0, 40, 110, 170, durationInFrames], [100, 50, 50, 22, 50], { ...clamp, easing: Easing.inOut(Easing.cubic) });
  const tag = (text: string, color: string, side: "left" | "right") => (
    <div style={{ position: "absolute", top: 56, [side]: 64, fontFamily: mono, fontSize: 26, letterSpacing: 3, color, background: "rgba(7,8,10,.72)", border: `1px solid ${color}`, borderRadius: 10, padding: "10px 18px", textTransform: "uppercase" }}>
      {text}
    </div>
  );
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Video src={staticFile("render/raw.mp4")} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      <AbsoluteFill style={{ clipPath: `inset(0 0 0 ${x}%)` }}>
        <Video src={staticFile("render/difix.mp4")} muted style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
      <div style={{ position: "absolute", top: 0, bottom: 0, left: `${x}%`, width: 3, background: C.text, boxShadow: "0 0 24px rgba(255,255,255,.5)" }} />
      {tag("raw splat", C.dim, "left")}
      <div style={{ opacity: x < 95 ? 1 : 0 }}>{tag("Difix · AI-enhanced", C.ai, "right")}</div>
      <Shade />
      <Caption kicker="--enhance difix" title="Floaters and smears, cleaned up." sub="NVIDIA Difix runs on every frame; the car and path are composited on top, never generated." />
    </AbsoluteFill>
  );
};

const Chase: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Clip src="render/difix.mp4" zoom={[1.0, 1.04]} trim={105} />
    <Shade />
    <Caption kicker="--cam chase" title="A chase cam that never existed." sub="Six car cameras, one rigid rig, a Gaussian splat trained on your Mac. Chase, driver, drone or free cam." />
  </AbsoluteFill>
);

const Cta: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center", textAlign: "center" }}>
    <Kicker>free · open source · runs locally</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 84, color: C.text, marginTop: 20 }}>
        Open Tesla <span style={{ color: C.accent }}>Reality View</span>
      </div>
    </Rise>
    <Rise delay={14}>
      <div style={{ fontFamily: mono, fontSize: 36, color: "#c9ced8", marginTop: 26 }}>github.com/jchacker5/open-tesla-reality-view</div>
    </Rise>
    <Rise delay={22}>
      <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 24, color: C.dim, marginTop: 40 }}>
        Difix weights: NVIDIA License, non-commercial use only.
      </div>
    </Rise>
  </AbsoluteFill>
);

export const RenderDemo: React.FC = () => {
  const order: [keyof typeof SCENES, React.FC][] = [
    ["title", Title],
    ["command", Command],
    ["compare", Compare],
    ["chase", Chase],
    ["cta", Cta],
  ];
  return (
    <TransitionSeries>
      {order.flatMap(([k, S], i) => {
        const seq = (
          <TransitionSeries.Sequence key={k} durationInFrames={SCENES[k]}>
            <S />
          </TransitionSeries.Sequence>
        );
        return i === 0 ? [seq] : [<TransitionSeries.Transition key={`${k}-t`} presentation={fade()} timing={linearTiming({ durationInFrames: T })} />, seq];
      })}
    </TransitionSeries>
  );
};
