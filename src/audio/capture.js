import { createWindower } from './ring-buffer.js';
import { WINDOW_SEC, HOP_SEC } from '../features/config.js';

export async function startCapture(onWindow) {
  // All three flags default to true and are all wrong here. AGC destroys the
  // amplitude reference; noise suppression is speech-tuned and attenuates the
  // exact broadband signature being hunted.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const ctx = new AudioContext();
  const sampleRate = ctx.sampleRate;
  await ctx.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'capture-processor');
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;

  const windower = createWindower(
    Math.round(WINDOW_SEC * sampleRate),
    Math.round(HOP_SEC * sampleRate),
    (samples) => onWindow(samples, sampleRate),
  );

  node.port.onmessage = (e) => windower.push(e.data);
  source.connect(analyser);
  source.connect(node);

  return {
    sampleRate,
    analyser,
    stop() {
      node.port.onmessage = null;
      node.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
  };
}
