import { startCapture } from './audio/capture.js';
import { extractFeatures, FEATURE_ORDER } from './features/extract.js';
import { classify, DEFAULT_WEIGHTS } from './classify/layer1.js';
import { createDetector } from './detect/state-machine.js';
import { createSpectrogram } from './ui/spectrogram.js';
import { createMeters } from './ui/meters.js';
import { createLog } from './ui/log.js';
import { createRecorder } from './ui/recorder.js';

const startButton = document.getElementById('start');
const verdict = document.getElementById('verdict');
const meters = createMeters(document.getElementById('meters'), FEATURE_ORDER);
const log = createLog(document.getElementById('log'));
const recorder = createRecorder({
  kindSelect: document.getElementById('rec-kind'),
  labelInput: document.getElementById('rec-label'),
  button: document.getElementById('rec'),
  elapsed: document.getElementById('rec-elapsed'),
});

let session = null;
let lastFeatures = { f0: 0 };
const startedAt = Date.now();

const detector = createDetector({
  onAlert: ({ at, confidence }) => log.add({ at, confidence, f0: lastFeatures.f0 }),
});

function onWindow(samples, sampleRate) {
  const features = extractFeatures(samples, sampleRate);
  lastFeatures = features;
  meters.update(features);

  const p = classify(features, DEFAULT_WEIGHTS);
  const state = detector.push(p, Date.now() - startedAt);

  verdict.textContent = state === 'detecting'
    ? `MULTIROTOR — confidence ${p.toFixed(2)}, f0 ${features.f0.toFixed(0)} Hz`
    : `Listening — ${p.toFixed(2)}`;
  verdict.className = state;
}

startButton.addEventListener('click', async () => {
  if (session) {
    session.stop();
    session = null;
    detector.reset();
    recorder.setListening(false);
    startButton.textContent = 'Start listening';
    verdict.textContent = 'Idle';
    verdict.className = '';
    return;
  }
  try {
    session = await startCapture(onWindow, recorder.onFrame);
    createSpectrogram(document.getElementById('spectrogram'), session.analyser);
    recorder.setListening(true, session.sampleRate);
    startButton.textContent = 'Stop';
  } catch (err) {
    verdict.textContent = `Microphone unavailable: ${err.message}`;
  }
});

if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');

if (!DEFAULT_WEIGHTS.trained) {
  verdict.textContent = 'Idle — classifier is untrained, scores are provisional';
}
