// Scrolling spectrogram driven by AnalyserNode. Display only - the analysis
// path uses the owned FFT so it can be matched in Python.
export function createSpectrogram(canvas, analyser) {
  const ctx = canvas.getContext('2d');
  const bins = new Uint8Array(analyser.frequencyBinCount);
  let raf = 0;

  function draw() {
    analyser.getByteFrequencyData(bins);
    ctx.drawImage(canvas, -1, 0);
    // Only the lowest quarter of the spectrum matters here; drone blade-pass
    // tones and their harmonics all sit below ~6 kHz.
    const shown = Math.floor(bins.length / 4);
    for (let i = 0; i < shown; i++) {
      const v = bins[i] / 255;
      const y = canvas.height - Math.round((i / shown) * canvas.height);
      ctx.fillStyle = `hsl(${200 - v * 160}, 90%, ${8 + v * 52}%)`;
      ctx.fillRect(canvas.width - 1, y, 1, Math.ceil(canvas.height / shown) + 1);
    }
    raf = requestAnimationFrame(draw);
  }

  draw();
  return { stop: () => cancelAnimationFrame(raf) };
}
