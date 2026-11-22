import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";
import { Audio } from "@remotion/media";
import type React from "react";
import { AbsoluteFill, Easing, getStaticFiles, interpolate, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig } from "remotion";

export const inter = loadInter("normal", { weights: ["300", "500", "700", "800"], subsets: ["latin"] }).fontFamily;
export const mono = loadMono("normal", { weights: ["400", "500"], subsets: ["latin"] }).fontFamily;
export const C = { bg: "#07080a", text: "#eef0f4", dim: "#8a909c", accent: "#3d7bff", line: "#1f232a" };
export const ease = Easing.bezier(0.16, 1, 0.3, 1);

export function useIn(delay = 0, dur = 24) {
  const f = useCurrentFrame();
  return interpolate(f - delay, [0, dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: ease });
}

export const Rise: React.FC<{ delay?: number; children: React.ReactNode; style?: React.CSSProperties }> = ({
  delay = 0,
  children,
  style,
}) => {
  const p = useIn(delay);
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 28}px)`, ...style }}>{children}</div>;
};

export const Kicker: React.FC<{ children: React.ReactNode; delay?: number; size?: number }> = ({
  children,
  delay = 0,
  size = 24,
}) => (
  <Rise delay={delay}>
    <div style={{ fontFamily: mono, fontSize: size, letterSpacing: size / 6, color: C.accent, textTransform: "uppercase" }}>
      {children}
    </div>
  </Rise>
);

export const Clip: React.FC<{ src: string; zoom?: [number, number]; trim?: number; style?: React.CSSProperties }> = ({
  src,
  zoom = [1.0, 1.06],
  trim = 0,
  style,
}) => {
  const f = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const s = interpolate(f, [0, durationInFrames], zoom);
  return (
    <AbsoluteFill style={{ transform: `scale(${s})`, ...style }}>
      <OffthreadVideo src={staticFile(src)} muted trimBefore={trim} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </AbsoluteFill>
  );
};

/** Voiceover if the file exists (renders silently until voice/*.mp3 is generated). */
export const Voice: React.FC<{ file: string }> = ({ file }) => {
  const have = getStaticFiles().some((f) => f.name === file);
  return have ? <Audio src={staticFile(file)} /> : null;
};
