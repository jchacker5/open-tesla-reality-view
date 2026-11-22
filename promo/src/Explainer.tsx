import type React from "react";
import { AbsoluteFill, interpolate, Sequence, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Clip, inter, Kicker, mono, Rise, Voice } from "./Shared";
import { VOICE } from "./voice";

/** 16:9 "how it works in 40 seconds": one step per spoken step, real commands, real footage. */
const STEPS = [
  { n: "01", t: "Copy your footage", cmd: "otrv ingest", d: "Reads the TeslaCam folder off the USB drive. Read-only, resumable.", src: "mosaic.mp4", w: 13 },
  { n: "02", t: "Pick 20 seconds", cmd: "otrv prep <clip> --mask-movers", d: "Frames from every camera; your car's body and moving traffic masked out.", src: "mosaic.mp4", w: 18 },
  { n: "03", t: "Solve the rig, train the splat", cmd: "otrv recon drive1", d: "COLMAP solves all cameras as one rigid rig. Brush trains a Gaussian splat on your Mac's GPU.", src: "orbit.mp4", w: 22 },
  { n: "04", t: "Replay it", cmd: "otrv view drive1", d: "Chase cam, driver view or free flight, with the raw feeds beside it.", src: "chase.mp4", w: 22 },
];

const Step: React.FC<{ s: (typeof STEPS)[number] }> = ({ s }) => {
  const f = useCurrentFrame();
  const typed = `$ ${s.cmd}`.slice(0, Math.max(0, Math.floor((f - 14) / 1.5)));
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <div style={{ position: "absolute", left: 820, top: 0, right: 0, bottom: 0, overflow: "hidden" }}>
        <Clip src={s.src} zoom={[1.02, 1.08]} trim={s.n === "02" ? 150 : 0} />
        <AbsoluteFill style={{ background: "linear-gradient(90deg, #07080a 0%, rgba(7,8,10,0) 30%)" }} />
      </div>
      <div style={{ position: "absolute", left: 96, top: 0, bottom: 0, width: 820, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <Kicker size={26}>step {s.n} / 04</Kicker>
        <Rise delay={4}>
          <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 68, color: C.text, lineHeight: 1.05, marginTop: 18 }}>{s.t}</div>
        </Rise>
        <div
          style={{
            fontFamily: mono,
            fontSize: 30,
            color: C.text,
            marginTop: 34,
            padding: "16px 22px",
            background: "#0e1014",
            border: `1px solid ${C.line}`,
            borderRadius: 12,
            minHeight: 40,
          }}
        >
          {typed}
          <span style={{ color: C.accent, opacity: Math.floor(f / 15) % 2 ? 0 : 1 }}>▍</span>
        </div>
        <Rise delay={12}>
          <div style={{ fontFamily: inter, fontWeight: 300, fontSize: 32, color: "#c9ced8", marginTop: 30, lineHeight: 1.35, maxWidth: 640 }}>{s.d}</div>
        </Rise>
      </div>
    </AbsoluteFill>
  );
};

const Intro: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center", textAlign: "center" }}>
    <AbsoluteFill style={{ opacity: 0.3, filter: "blur(12px)" }}>
      <Clip src="chase.mp4" zoom={[1.2, 1.3]} />
    </AbsoluteFill>
    <Kicker>open tesla reality view</Kicker>
    <Rise delay={6}>
      <div style={{ fontFamily: inter, fontWeight: 700, fontSize: 96, color: C.text, marginTop: 20, letterSpacing: -2 }}>
        Dashcam → 3D, in four steps
      </div>
    </Rise>
  </AbsoluteFill>
);

export const Explainer: React.FC = () => {
  const { durationInFrames } = useVideoConfig();
  const f = useCurrentFrame();
  const intro = 120;
  const outro = 90;
  const body = durationInFrames - intro - outro;
  const sum = STEPS.reduce((a, s) => a + s.w, 0);
  let t = intro;
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Sequence durationInFrames={intro}>
        <Intro />
      </Sequence>
      {STEPS.map((s) => {
        const d = Math.round((body * s.w) / sum);
        const from = t;
        t += d;
        return (
          <Sequence key={s.n} from={from} durationInFrames={d} premountFor={30}>
            <Step s={s} />
          </Sequence>
        );
      })}
      <Sequence from={durationInFrames - outro}>
        <AbsoluteFill style={{ background: C.bg, alignItems: "center", justifyContent: "center" }}>
          <Rise>
            <div style={{ fontFamily: mono, fontSize: 40, color: C.text, padding: "18px 30px", border: `1px solid ${C.accent}`, borderRadius: 14 }}>
              github.com/jchacker5/open-tesla-reality-view
            </div>
          </Rise>
        </AbsoluteFill>
      </Sequence>
      <div style={{ position: "absolute", bottom: 0, left: 0, height: 6, width: `${(f / durationInFrames) * 100}%`, background: C.accent, opacity: interpolate(f, [0, 10], [0, 1]) }} />
      <Voice file={VOICE.explainer.file} />
    </AbsoluteFill>
  );
};
