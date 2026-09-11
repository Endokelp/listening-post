// Encodes captured frames exactly as the detector heard them (no processing,
// no re-resampling) so field recordings match the model's input.

export function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const buf = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buf);
  const str = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };

  str(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate = sampleRate * blockAlign
  view.setUint16(32, 2, true); // block align (1 channel * 16 bits / 8)
  view.setUint16(34, 16, true); // bits per sample
  str(36, 'data');
  view.setUint32(40, length * 2, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const clamped = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, Math.round(clamped * 0x7fff), true);
      offset += 2;
    }
  }
  return buf;
}

// Filename contract with the Python side: split("__") -> [kind, source, label, recording].
// source is always 'own' (this device's mic, vs. downloaded/third-party clips).
export function recordingName(kind, label, date) {
  const clean = label.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/^-+|-+$/g, '') || 'unlabeled';
  const pad = (n) => String(n).padStart(2, '0');
  const stamp = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
  return `${kind}__own__${clean}__${stamp}.wav`;
}
