export function createLog(table) {
  const entries = [];
  table.innerHTML = '<thead><tr><th>Time</th><th>Confidence</th><th>f0 (Hz)</th></tr></thead><tbody></tbody>';
  const body = table.querySelector('tbody');

  return {
    add({ at, confidence, f0 }) {
      const entry = { at, confidence, f0, wallClock: new Date().toISOString() };
      entries.push(entry);
      const row = body.insertRow(0);
      row.insertCell().textContent = new Date().toLocaleTimeString();
      row.insertCell().textContent = confidence.toFixed(2);
      row.insertCell().textContent = f0.toFixed(0);
    },
    toCsv() {
      return ['wallClock,elapsedMs,confidence,f0']
        .concat(entries.map((e) => `${e.wallClock},${e.at},${e.confidence.toFixed(4)},${e.f0.toFixed(1)}`))
        .join('\n');
    },
    get entries() {
      return [...entries];
    },
  };
}
