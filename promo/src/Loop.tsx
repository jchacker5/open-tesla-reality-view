import type React from "react";
import { AbsoluteFill } from "remotion";
import { C, Clip, mono } from "./Shared";

/** 1080x1080 silent split-screen loop: built for autoplay-muted feeds. Same moment, two worlds. */
const Label: React.FC<{ children: React.ReactNode; top: number }> = ({ children, top }) => (
  <div
    style={{
      position: "absolute",
      left: 28,
      top,
      fontFamily: mono,
      fontSize: 26,
      letterSpacing: 4,
      color: C.text,
      background: "rgba(7,8,10,.7)",
      padding: "8px 14px",
      borderRadius: 8,
    }}
  >
    {children}
  </div>
);

export const Loop: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 538, overflow: "hidden" }}>
      <Clip src="mosaic.mp4" zoom={[1.0, 1.0]} />
    </div>
    <div style={{ position: "absolute", left: 0, right: 0, top: 542, height: 538, overflow: "hidden" }}>
      <Clip src="chase.mp4" zoom={[1.0, 1.0]} />
    </div>
    <div style={{ position: "absolute", left: 0, right: 0, top: 538, height: 4, background: C.accent }} />
    <Label top={28}>DASHCAM · 4 CAMERAS</Label>
    <Label top={570}>
      3D REPLAY · <span style={{ color: C.accent }}>GAUSSIAN SPLAT</span>
    </Label>
    <div style={{ position: "absolute", right: 28, bottom: 26, fontFamily: mono, fontSize: 22, color: C.text, opacity: 0.85 }}>
      open-tesla-reality-view
    </div>
  </AbsoluteFill>
);
