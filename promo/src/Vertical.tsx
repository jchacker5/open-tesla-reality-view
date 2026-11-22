import type React from "react";
import { AbsoluteFill, interpolate, Sequence, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Clip, inter, mono, Voice } from "./Shared";
import { LINES, lineTimes, VOICE } from "./voice";

/** 1080x1920 hook-first short for X / Reels / Shorts: footage changes with every spoken line. */
const Caption: React.FC<{ text: string }> = ({ text }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const words = text.split(" ");
  return (
    <div
      style={{
        position: "absolute",
        left: 70,
        right: 70,
        top: 1180,
        textAlign: "center",
        fontFamily: inter,
        fontWeight: 800,
        fontSize: 84,
        lineHeight: 1.08,
        color: C.text,
        letterSpacing: -1.5,
        textShadow: "0 6px 30px rgba(0,0,0,.85), 0 2px 6px rgba(0,0,0,.9)",
      }}
    >
      {words.map((w, i) => {
        const s = spring({ frame: f - i * 3, fps, config: { damping: 14, stiffness: 180 } });
        return (
          <span
            key={`${w}-${i}`}
            style={{ display: "inline-block", marginRight: 22, transform: `scale(${0.6 + 0.4 * s})`, opacity: s }}
          >
            {/3D|rig|splat|angle|open/i.test(w) ? <span style={{ color: C.accent }}>{w}</span> : w}
          </span>
        );
      })}
    </div>
  );
};

// Which footage sits behind each spoken line.
const SHOTS: { src: string; zoom: [number, number]; trim?: number }[] = [
  { src: "mosaic.mp4", zoom: [1.0, 1.08] },
  { src: "chase.mp4", zoom: [1.25, 1.1] },
  { src: "mosaic.mp4", zoom: [1.05, 1.0], trim: 120 },
  { src: "chase.mp4", zoom: [1.0, 1.08], trim: 150 },
  { src: "orbit.mp4", zoom: [1.0, 1.1] },
  { src: "orbit.mp4", zoom: [1.1, 1.2], trim: 120 },
  { src: "chase.mp4", zoom: [1.05, 1.15], trim: 300 },
];

const Flash: React.FC = () => {
  const f = useCurrentFrame();
  return <AbsoluteFill style={{ background: "#fff", opacity: interpolate(f, [0, 6], [0.5, 0], { extrapolateRight: "clamp" }) }} />;
};

export const Vertical: React.FC = () => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const lines = lineTimes(LINES.short, durationInFrames);
  const last = lines[lines.length - 1];
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      {lines.map((l, i) => (
        <Sequence key={l.text} from={l.from} durationInFrames={i === lines.length - 1 ? durationInFrames - l.from : l.dur} premountFor={30}>
          <Clip src={SHOTS[i].src} zoom={SHOTS[i].zoom} trim={SHOTS[i].trim ?? 0} />
          <AbsoluteFill
            style={{ background: "linear-gradient(180deg, rgba(7,8,10,.6) 0%, rgba(7,8,10,0) 22%, rgba(7,8,10,0) 50%, rgba(7,8,10,.75) 100%)" }}
          />
          {i > 0 && <Flash />}
          <Caption text={l.text} />
        </Sequence>
      ))}
      <Sequence durationInFrames={lines[0].from}>
        <Clip src="mosaic.mp4" zoom={[1.0, 1.0]} />
      </Sequence>

      {/* end card over the last line */}
      <Sequence from={last.from} premountFor={30}>
        <EndCard />
      </Sequence>

      {/* brand + progress bar: keeps viewers oriented and watching */}
      <div style={{ position: "absolute", top: 90, left: 0, right: 0, textAlign: "center" }}>
        <div style={{ fontFamily: mono, fontSize: 30, letterSpacing: 6, color: C.text, opacity: 0.9 }}>
          OPEN TESLA <span style={{ color: C.accent }}>REALITY VIEW</span>
        </div>
      </div>
      <div style={{ position: "absolute", top: 0, left: 0, height: 10, width: `${(f / durationInFrames) * 100}%`, background: C.accent }} />
      <Voice file={VOICE.short.file} />
    </AbsoluteFill>
  );
};

const EndCard: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: f - 8, fps, config: { damping: 200 } });
  return (
    <div style={{ position: "absolute", left: 70, right: 70, top: 1560, textAlign: "center", opacity: s, transform: `translateY(${(1 - s) * 30}px)` }}>
      <div
        style={{
          display: "inline-block",
          fontFamily: mono,
          fontSize: 31,
          color: C.text,
          padding: "18px 26px",
          border: `2px solid ${C.accent}`,
          borderRadius: 18,
          background: "rgba(7,8,10,.75)",
        }}
      >
        github.com/jchacker5/open-tesla-reality-view
      </div>
      <div style={{ fontFamily: inter, fontSize: 24, color: C.dim, marginTop: 18 }}>Free · open source · not affiliated with Tesla</div>
    </div>
  );
};
