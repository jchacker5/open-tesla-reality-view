import { Composition } from "remotion";
import { Explainer } from "./Explainer";
import { Loop } from "./Loop";
import { FPS, Promo, PROMO_FRAMES } from "./Promo";
import { Vertical } from "./Vertical";
import { VOICE } from "./voice";

const frames = (s: number, pad = 1.0) => Math.ceil((s + pad) * FPS);

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Promo" component={Promo} durationInFrames={PROMO_FRAMES} fps={FPS} width={1920} height={1080} />
    <Composition id="Short" component={Vertical} durationInFrames={frames(VOICE.short.seconds, 1.5)} fps={FPS} width={1080} height={1920} />
    <Composition id="Explainer" component={Explainer} durationInFrames={frames(VOICE.explainer.seconds, 6)} fps={FPS} width={1920} height={1080} />
    <Composition id="Loop" component={Loop} durationInFrames={12 * FPS} fps={FPS} width={1080} height={1080} />
  </>
);
