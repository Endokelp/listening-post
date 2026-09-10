const LABELS = {
  harmonicStack: 'Harmonic stack',
  rotorBeat: 'Rotor beat',
  bandRatio: 'Band energy',
  spectralFlatness: 'Flatness',
};

export function createMeters(container, featureOrder) {
  const bars = {};
  container.innerHTML = '';
  for (const key of featureOrder) {
    const row = document.createElement('div');
    row.className = 'meter';
    const label = document.createElement('span');
    label.textContent = LABELS[key] ?? key;
    const track = document.createElement('div');
    track.className = 'meter-track';
    const fill = document.createElement('div');
    fill.className = 'meter-fill';
    const value = document.createElement('span');
    value.className = 'meter-value';
    track.append(fill);
    row.append(label, track, value);
    container.append(row);
    bars[key] = { fill, value };
  }

  return {
    update(features) {
      for (const [key, { fill, value }] of Object.entries(bars)) {
        const v = Math.max(0, Math.min(1, features[key] ?? 0));
        fill.style.width = `${(v * 100).toFixed(0)}%`;
        value.textContent = v.toFixed(2);
      }
    },
  };
}
