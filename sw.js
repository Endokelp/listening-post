// Network first, so a redeploy shows up immediately. The cache only answers
// when there is no signal - a listening post is often somewhere without one.
// ponytail: runtime caching only, no precache list. Every file is cached the
// first time it loads online (the worklet loads on the first "Start
// listening"), so open the app and start it once before going offline.
const CACHE = 'listening-post';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res.ok) {
          const copy = res.clone();
          e.waitUntil(caches.open(CACHE).then((c) => c.put(e.request, copy)));
        }
        return res;
      })
      .catch(() => caches.match(e.request)),
  );
});
