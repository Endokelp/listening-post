import { encodeWav, recordingName } from '../audio/wav.js';

// Field-recording tool: saves the exact frames the detector sees (see
// capture.js) so drone/confuser clips can be reused as training data.
export function createRecorder({ kindSelect, labelInput, button, elapsed }) {
  let frames = null; // null when not recording
  let sampleRate = 0;
  let startedAt = 0;
  let timer = 0;

  function tick() {
    elapsed.textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
  }

  function stop() {
    clearInterval(timer);
    if (frames && frames.length) {
      const buf = encodeWav(frames, sampleRate);
      const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = recordingName(kindSelect.value, labelInput.value, new Date());
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000); // revoke after the download has actually started
    }
    frames = null;
    button.textContent = 'Record';
    elapsed.textContent = '';
  }

  button.addEventListener('click', () => {
    if (frames) { stop(); return; }
    frames = [];
    startedAt = Date.now();
    button.textContent = 'Stop & save';
    timer = setInterval(tick, 500);
    tick();
  });

  return {
    onFrame(frame) {
      if (frames) frames.push(new Float32Array(frame)); // copy: worklet buffer may be reused
    },
    setListening(listening, rate) {
      if (rate) sampleRate = rate;
      button.disabled = !listening;
      if (!listening && frames) stop();
    },
  };
}
