// Social cuts around `otrv render`, using the same footage as RenderDemo (public/render/*.mp4):
//   RenderShort  1080x1920  raw on top, Difix below, in sync: the phone-timeline version
//   StackClip    1920x1080  how it works, with the measured numbers from the demo run
//   ChaseLoop    1080x1080  seamless chase-cam loop (source for the GIF)
// Numbers in StackClip come from that run (`otrv bench demo`, Apple M4 32 GB) and cpp/README.md.
import { Video } from "@remotion/media";
import { linearTiming, TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import type React from "react";
import { AbsoluteFill, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, clamp, ease, inter, Kicker, mono, Rise } from "./RenderDemo";

const T = 15;

const series = (scenes: [number, React.FC][]) => (
  <TransitionSeries>
    {scenes.flatMap(([d, S], i) => {
      const seq = (
        <TransitionSeries.Sequence key={i} durationInFrames={d}>
          <S />
        </TransitionSeries.Sequence>
      );
      return i === 0 ? [seq] : [<TransitionSeries.Transition key={`t${i}`} presentation={fade()} timing={linearTiming({ durationInFrames: T })} />, seq];
    })}
  </TransitionSeries>
);
const total = (d: number[]) => d.reduce((a, b) => a + b, 0) - T * (d.length - 1);

const Tag: React.FC<{ text: string; color: string; style?: React.CSSProperties }> = ({ text, color, style }) => (
  <div style={{ position: "absolute", fontFamily: mono, fontSize: 30, letterSpacing: 3, color, background: "rgba(7,8,10,.75)", border: `1px solid ${color}`, borderRadius: 10, padding: "10px 18px", textTransform: "uppercase", ...style }}>
    {text}
  </div>
);

// ---- RenderShort (9:16) ------------------------------------------------------------------------

const ShortHook: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center", textAlign: "center", padding: 80 }}>
    <Kicker>tesla dashcam → 3D</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 96, lineHeight: 1.05, color: C.text, marginTop: 28 }}>
        Same scene.
        <br />
        <span style={{ color: C.ai }}>One flag.</span>
      </div>
    </Rise>
    <Rise delay={16}>
      <div style={{ fontFamily: mono, fontSize: 40, color: "#c9ced8", marginTop: 40 }}>--enhance difix</div>
    </Rise>
  </AbsoluteFill>
);

const ShortStack: React.FC = () => {
  const f = useCurrentFrame();
  const line = interpolate(f, [0, 20], [0, 1], { ...clamp, easing: ease });
  const half: React.CSSProperties = { position: "absolute", left: 0, width: "100%", height: "50%", overflow: "hidden" };
  // 16:9 source in a 1080x960 half: fill the height and centre-crop the sides (objectFit isn't applied
  // to @remotion/media's <Video>, so size it explicitly).
  const w = (960 * 16) / 9;
  const vid: React.CSSProperties = { position: "absolute", height: 960, width: w, left: (1080 - w) / 2, top: 0 };
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <div style={{ ...half, top: 0 }}>
        <Video src={staticFile("render/raw.mp4")} muted style={vid} />
        <Tag text="raw splat" color={C.dim} style={{ top: 48, left: 48 }} />
      </div>
      <div style={{ ...half, top: "50%" }}>
        <Video src={staticFile("render/difix.mp4")} muted style={vid} />
        <Tag text="Difix · AI-enhanced" color={C.ai} style={{ top: 48, left: 48 }} />
      </div>
      <div style={{ position: "absolute", top: "50%", left: 0, height: 4, width: `${line * 100}%`, background: C.text, boxShadow: "0 0 24px rgba(255,255,255,.5)" }} />
    </AbsoluteFill>
  );
};

const ShortCta: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center", textAlign: "center", padding: 80 }}>
    <Kicker>open source · runs on a mac</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: mono, fontWeight: 500, fontSize: 92, color: C.text, marginTop: 26 }}>
        otrv <span style={{ color: C.accent }}>render</span>
      </div>
    </Rise>
    <Rise delay={14}>
      <div style={{ fontFamily: mono, fontSize: 34, color: "#c9ced8", marginTop: 34, lineHeight: 1.5 }}>
        github.com/jchacker5/
        <br />
        open-tesla-reality-view
      </div>
    </Rise>
    <Rise delay={22}>
      <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 26, color: C.dim, marginTop: 44 }}>Difix: NVIDIA License, non-commercial.</div>
    </Rise>
  </AbsoluteFill>
);

const SHORT: [number, React.FC][] = [
  [75, ShortHook],
  [300, ShortStack],
  [90, ShortCta],
];
export const SHORT_FRAMES = total(SHORT.map(([d]) => d));
export const RenderShort: React.FC = () => series(SHORT);

// ---- StackClip (16:9) --------------------------------------------------------------------------

const STAGES = [
  { k: "ingest", t: "6 cameras", d: "TeslaCam clips + SEI telemetry", stat: "720 frames @ 6 fps" },
  { k: "sfm", t: "COLMAP rig", d: "global mapper, cameras as one rig", stat: "0.38 px reprojection" },
  { k: "train", t: "Brush · Metal", d: "Gaussian splats on the M4 GPU", stat: "207k splats · 10 min" },
  { k: "native", t: "C++20 core", d: "SEI decoder, GPS georef, nanobind", stat: "2.9x faster than Python" },
  { k: "render", t: "Playwright", d: "headless Chromium on Metal", stat: "1.5 frames/s capture" },
  { k: "enhance", t: "NVIDIA Difix", d: "one-step diffusion, MPS fp16", stat: "flow-guided anti-flicker" },
];

const StackTitle: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, padding: "0 120px", justifyContent: "center" }}>
    <Kicker>under the hood</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 88, color: C.text, marginTop: 20, lineHeight: 1.05 }}>
        From a USB stick
        <br />
        to a <span style={{ color: C.accent }}>3D film</span>, on a laptop.
      </div>
    </Rise>
    <Rise delay={16}>
      <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 36, color: "#c9ced8", marginTop: 26 }}>
        Python · C++20 · TypeScript · Rust (Brush) · Metal. No cloud.
      </div>
    </Rise>
  </AbsoluteFill>
);

const StackCards: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.bg, padding: "90px 96px", justifyContent: "center" }}>
      <Kicker>the pipeline · measured on an apple m4</Kicker>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 26, marginTop: 44 }}>
        {STAGES.map((s, i) => {
          const p = interpolate(f - 8 - i * 12, [0, 22], [0, 1], { ...clamp, easing: ease });
          const lit = f > 8 + i * 12 + 18;
          return (
            <div key={s.k} style={{ opacity: p, transform: `translateY(${(1 - p) * 40}px)`, border: `1px solid ${lit ? C.accent : C.line}`, background: "#0e1014", borderRadius: 18, padding: "30px 30px", boxShadow: lit ? "0 0 40px rgba(61,123,255,.16)" : "none" }}>
              <div style={{ fontFamily: mono, fontSize: 22, color: C.accent }}>
                {String(i + 1).padStart(2, "0")} · {s.k}
              </div>
              <div style={{ fontFamily: inter, fontWeight: 500, fontSize: 40, color: C.text, marginTop: 14 }}>{s.t}</div>
              <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 25, color: C.dim, marginTop: 8 }}>{s.d}</div>
              <div style={{ fontFamily: mono, fontSize: 24, color: C.ai, marginTop: 18 }}>{s.stat}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const LESSONS = [
  ["Low-res footage split SfM into 16 pieces", "global mapper: 24 → 450 frames registered"],
  ["One frame crashed the splat trainer", "bisected 450 frames to the one Brush rejects"],
  ["Two decoders, one behaviour", "C++ checked against Python on ~7,000 fuzzed inputs"],
  ["Every run is scored", "otrv bench gates and logs each reconstruction"],
];

const StackLessons: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: C.bg, padding: "0 120px", justifyContent: "center" }}>
      <Kicker>what building it taught me</Kicker>
      <div style={{ marginTop: 40 }}>
        {LESSONS.map(([a, b], i) => {
          const p = interpolate(f - 6 - i * 16, [0, 20], [0, 1], { ...clamp, easing: ease });
          return (
            <div key={a} style={{ opacity: p, transform: `translateX(${(1 - p) * -30}px)`, display: "flex", alignItems: "baseline", gap: 28, marginTop: 30 }}>
              <div style={{ fontFamily: mono, fontSize: 28, color: C.accent, width: 50 }}>{String(i + 1).padStart(2, "0")}</div>
              <div>
                <div style={{ fontFamily: inter, fontWeight: 500, fontSize: 46, color: C.text }}>{a}</div>
                <div style={{ fontFamily: mono, fontSize: 28, color: C.ai, marginTop: 6 }}>{b}</div>
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const StackEnd: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <AbsoluteFill style={{ opacity: 0.35 }}>
      <Video src={staticFile("render/difix.mp4")} muted trimBefore={60} style={{ position: "absolute", width: 1920, height: 1080 }} />
    </AbsoluteFill>
    <AbsoluteFill style={{ background: "radial-gradient(ellipse at center, rgba(7,8,10,.35), rgba(7,8,10,.95) 75%)" }} />
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", textAlign: "center" }}>
      <Kicker>mit · 380+ tests · ci on linux + macos</Kicker>
      <Rise delay={6}>
        <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 84, color: C.text, marginTop: 20 }}>
          Open Tesla <span style={{ color: C.accent }}>Reality View</span>
        </div>
      </Rise>
      <Rise delay={14}>
        <div style={{ fontFamily: mono, fontSize: 36, color: "#c9ced8", marginTop: 26 }}>github.com/jchacker5/open-tesla-reality-view</div>
      </Rise>
    </AbsoluteFill>
  </AbsoluteFill>
);

const STACK: [number, React.FC][] = [
  [105, StackTitle],
  [210, StackCards],
  [195, StackLessons],
  [120, StackEnd],
];
export const STACK_FRAMES = total(STACK.map(([d]) => d));
export const StackClip: React.FC = () => series(STACK);

// ---- ChaseLoop (1:1) ---------------------------------------------------------------------------
// The first and last 12 frames cross-fade, so the GIF loops without a jump.

export const LOOP_FRAMES = 180;
export const ChaseLoop: React.FC = () => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const X = 12;
  const tail = interpolate(f, [durationInFrames - X, durationInFrames], [0, 1], clamp);
  const w = (1080 * 16) / 9;
  const vid: React.CSSProperties = { position: "absolute", height: 1080, width: w, left: (1080 - w) / 2, top: 0 };
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      {/* source frames X..X+duration, then source frames 0..X fade in over the end: last frame = first */}
      <Video src={staticFile("render/difix.mp4")} muted trimBefore={X} style={vid} />
      <Sequence from={durationInFrames - X}>
        <AbsoluteFill style={{ opacity: tail }}>
          <Video src={staticFile("render/difix.mp4")} muted style={vid} />
        </AbsoluteFill>
      </Sequence>
      <div style={{ position: "absolute", left: 48, bottom: 44, fontFamily: mono, fontSize: 30, color: C.text, background: "rgba(7,8,10,.7)", borderRadius: 10, padding: "10px 18px" }}>
        otrv render --enhance difix
      </div>
    </AbsoluteFill>
  );
};

// ---- CompareGif (16:9, small) ------------------------------------------------------------------
// Raw splat on the left half, Difix on the right half of the same frame: the README / docs GIF.

export const COMPARE_FRAMES = 240;
export const CompareGif: React.FC = () => {
  const vid: React.CSSProperties = { position: "absolute", width: 960, height: 540, left: 0, top: 0 };
  const tag: React.CSSProperties = { top: 18, fontSize: 15, padding: "6px 10px", letterSpacing: 2, borderRadius: 7 };
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Video src={staticFile("render/raw.mp4")} muted style={vid} />
      <AbsoluteFill style={{ clipPath: "inset(0 0 0 50%)" }}>
        <Video src={staticFile("render/difix.mp4")} muted style={vid} />
      </AbsoluteFill>
      <div style={{ position: "absolute", top: 0, bottom: 0, left: 479, width: 2, background: C.text }} />
      <Tag text="raw splat" color={C.dim} style={{ ...tag, left: 18 }} />
      <Tag text="--enhance difix · AI" color={C.ai} style={{ ...tag, right: 18 }} />
    </AbsoluteFill>
  );
};
