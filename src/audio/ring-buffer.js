export function createRingBuffer(capacity) {
  const buf = new Float64Array(capacity);
  let start = 0, count = 0;

  return {
    write(chunk) {
      for (let i = 0; i < chunk.length; i++) {
        buf[(start + count) % capacity] = chunk[i];
        if (count < capacity) count++;
        else start = (start + 1) % capacity; // full: drop the oldest
      }
    },
    read(n) {
      const take = Math.min(n, count);
      const out = new Float64Array(take);
      for (let i = 0; i < take; i++) out[i] = buf[(start + i) % capacity];
      start = (start + take) % capacity;
      count -= take;
      return out;
    },
    get available() {
      return count;
    },
  };
}

// Emits fixed-size overlapping windows. Keeps windowSamples - hopSamples of
// history so consecutive windows overlap rather than abut.
export function createWindower(windowSamples, hopSamples, onWindow) {
  let pending = new Float64Array(0);

  return {
    push(chunk) {
      const merged = new Float64Array(pending.length + chunk.length);
      merged.set(pending, 0);
      for (let i = 0; i < chunk.length; i++) merged[pending.length + i] = chunk[i];
      pending = merged;

      let offset = 0;
      while (pending.length - offset >= windowSamples) {
        onWindow(pending.slice(offset, offset + windowSamples));
        offset += hopSamples;
      }
      pending = pending.slice(offset);
    },
  };
}
