import { Composition } from "remotion";
import { COMPARE_FRAMES, CompareGif, DroneLoop, LOOP_FRAMES, RenderShort, SHORT_FRAMES, STACK_FRAMES, StackClip } from "./Clips";
import { FPS, Promo, PROMO_FRAMES } from "./Promo";
import { RENDER_FPS, RENDER_FRAMES, RenderDemo } from "./RenderDemo";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Promo" component={Promo} durationInFrames={PROMO_FRAMES} fps={FPS} width={1920} height={1080} />
    <Composition id="RenderDemo" component={RenderDemo} durationInFrames={RENDER_FRAMES} fps={RENDER_FPS} width={1920} height={1080} />
    <Composition id="RenderShort" component={RenderShort} durationInFrames={SHORT_FRAMES} fps={RENDER_FPS} width={1080} height={1920} />
    <Composition id="StackClip" component={StackClip} durationInFrames={STACK_FRAMES} fps={RENDER_FPS} width={1920} height={1080} />
    <Composition id="DroneLoop" component={DroneLoop} durationInFrames={LOOP_FRAMES} fps={RENDER_FPS} width={1080} height={1080} />
    <Composition id="CompareGif" component={CompareGif} durationInFrames={COMPARE_FRAMES} fps={RENDER_FPS} width={960} height={540} />
  </>
);
