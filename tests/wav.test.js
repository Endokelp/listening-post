import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeWav, recordingName } from '../src/audio/wav.js';

function readStr(view, offset, len) {
  let s = '';
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

test('encodeWav writes a standard 44-byte PCM16 mono header', () => {
  const chunks = [Float32Array.from([0, 0.5, -1, 2.0])];
  const buf = encodeWav(chunks, 48000);
  const view = new DataView(buf);

  assert.equal(readStr(view, 0, 4), 'RIFF');
  assert.equal(readStr(view, 8, 4), 'WAVE');
  assert.equal(readStr(view, 12, 4), 'fmt ');
  assert.equal(readStr(view, 36, 4), 'data');

  const dataSize = 2 * chunks[0].length;
  assert.equal(view.getUint32(4, true), 36 + dataSize); // RIFF size
  assert.equal(view.getUint16(20, true), 1); // audio format: PCM
  assert.equal(view.getUint16(22, true), 1); // channels: mono
  assert.equal(view.getUint32(24, true), 48000); // sample rate
  assert.equal(view.getUint32(28, true), 48000 * 2); // byte rate
  assert.equal(view.getUint16(32, true), 2); // block align
  assert.equal(view.getUint16(34, true), 16); // bits per sample
  assert.equal(view.getUint32(40, true), dataSize);
  assert.equal(buf.byteLength, 44 + dataSize);
});

test('encodeWav clamps to [-1, 1] and scales with 0x7fff', () => {
  const buf = encodeWav([Float32Array.from([0, 0.5, -1, 2.0])], 48000);
  const view = new DataView(buf);
  const samples = [0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true));
  assert.deepEqual(samples, [0, 16384, -32767, 32767]);
});

test('encodeWav concatenates multiple chunks in order', () => {
  const buf = encodeWav([Float32Array.from([0, 0.5]), Float32Array.from([-1, 2.0])], 48000);
  const view = new DataView(buf);
  const samples = [0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true));
  assert.deepEqual(samples, [0, 16384, -32767, 32767]);
  assert.equal(view.getUint32(40, true), 2 * 4);
});

test('recordingName sanitizes the label and formats the stamp', () => {
  const date = new Date(2026, 8, 10, 7, 5, 9); // 2026-09-10 07:05:09 local
  const name = recordingName('drone', 'DJI 25m Hover!!', date);
  assert.equal(name, 'drone__own__dji-25m-hover__20260910-070509.wav');
});

test('recordingName collapses runs of invalid characters and trims edges', () => {
  const date = new Date(2026, 0, 1, 0, 0, 0);
  assert.equal(recordingName('confuser', '  ___leaf   blower--- ', date), 'confuser__own__leaf-blower__20260101-000000.wav');
});

test('recordingName falls back to "unlabeled" for an empty/invalid label', () => {
  const date = new Date(2026, 0, 1, 0, 0, 0);
  assert.equal(recordingName('background', '   ', date), 'background__own__unlabeled__20260101-000000.wav');
  assert.equal(recordingName('background', '___', date), 'background__own__unlabeled__20260101-000000.wav');
});

test('recordingName can never contain a "__" inside the sanitized label, so split("__") gives exactly 4 fields', () => {
  const date = new Date(2026, 5, 15, 12, 30, 45);
  const name = recordingName('drone', 'a__b  c__DJI__mini', date);
  const fields = name.replace(/\.wav$/, '').split('__');
  assert.equal(fields.length, 4);
  assert.deepEqual(fields, ['drone', 'own', 'a-b-c-dji-mini', '20260615-123045']);
});

test('recordingName stamp is YYYYMMDD-HHMMSS local time', () => {
  const date = new Date(2026, 8, 10, 7, 5, 9);
  const name = recordingName('drone', 'x', date);
  const stamp = name.split('__')[3].replace('.wav', '');
  assert.match(stamp, /^\d{8}-\d{6}$/);
  assert.equal(stamp, '20260910-070509');
});
