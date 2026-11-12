// Figures for the all-reduce post: log-log charts, per-machine bars, a topology
// diagram and an animated ring all-reduce. Each <figure class="ar-fig" data-fig="...">
// in the post is filled in by the matching function below.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const C = { nccl: 'var(--ar-nccl)', ring: 'var(--ar-ring)', fused: 'var(--ar-fused)', direct: 'var(--ar-direct)', copy: 'var(--ar-copy)', rr: 'var(--ar-rr)' };

  // ---------------------------------------------------------------- data (median us per call, from the CSVs)
  const S9 = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456];
  const S6 = [4096, 65536, 1048576, 16777216, 268435456, 1073741824];
  const S10 = [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824];
  const CHARTS = {
    t4: { sizes: S9, series: [
      { name: 'copy, serialised', color: C.copy, v: [32.9, 34.1, 54.4, 91.2, 250.9, 937.8, 3623.1, 14384.8, 57428.5] },
      { name: 'copy, async', color: C.ring, v: [19.4, 20.9, 28.2, 73.4, 230.5, 917.2, 3605.1, 14363.4, 57386.1] },
      { name: 'remote read in kernel', color: C.fused, v: [13.6, 14.9, 27.6, 80.7, 293.3, 1147.9, 4563.0, 18224.7, 72874.8] },
      { name: 'ring', color: C.direct, v: [22.4, 23.4, 34.7, 77.7, 264.0, 1009.6, 3989.2, 15901.2, 63544.5] },
      { name: 'NCCL', color: C.nccl, v: [28.4, 34.4, 59.9, 113.3, 308.1, 1087.2, 4206.4, 16713.2, 66675.3] }] },
    l4: { sizes: S9, series: [
      { name: 'copy all inputs, then sum', color: C.copy, v: [120.5, 120.8, 155.5, 248.8, 761.7, 2275.1, 8611.1, 33984.9, 150707.3] },
      { name: 'ring, copies', color: C.ring, v: [238.6, 236.9, 238.5, 255.1, 426.2, 1150.4, 2964.0, 10724.8, 41778.7] },
      { name: 'NCCL', color: C.nccl, v: [42.1, 46.2, 68.1, 143.1, 350.3, 1239.3, 4687.6, 18735.4, 73529.5] }] },
    a100h: { sizes: S9, series: [
      { name: 'copy all inputs, then sum', color: C.copy, v: [368.7, 367.5, 362.4, 372.7, 388.1, 501.6, 1270.9, 4110.7, 15213.9] },
      { name: 'read all inputs remotely', color: C.direct, v: [45.4, 45.5, 45.1, 45.3, 83.5, 184.9, 574.1, 2162.3, 8439.1] },
      { name: 'ring, copies', color: C.ring, v: [1148.6, 1153.2, 1152.1, 1148.1, 1153.3, 1158.4, 1165.0, 1165.9, 2972.2] },
      { name: 'ring, remote-read reduce', color: C.rr, v: [821.2, 826.9, 825.5, 824.1, 828.1, 830.8, 833.7, 810.8, 2269.7] },
      { name: 'NCCL', color: C.nccl, v: [103.1, 106.7, 110.6, 113.0, 130.0, 182.2, 318.9, 715.6, 2320.9] }] },
    sm: { sizes: S6, series: [
      { name: 'scalar kernel, full grid', color: C.copy, dash: true, v: [722.1, 724.2, 724.3, 724.9, 2213.7, 7847.3] },
      { name: 'scalar kernel, 24 x 512', color: C.copy, v: [723.2, 727.0, 725.4, 662.9, 6418.3, 24526.5] },
      { name: 'float4 kernel, 24 x 512', color: C.rr, v: [725.5, 727.1, 727.9, 728.6, 2237.2, 7906.3] },
      { name: 'NCCL', color: C.nccl, v: [102.7, 110.9, 129.0, 317.6, 2331.9, 8379.7] }] },
    fused: { sizes: S6, series: [
      { name: 'ring, host-driven', color: C.rr, v: [839.6, 838.0, 831.9, 830.5, 2253.2, 7925.9] },
      { name: 'fused, fence per thread', color: C.ring, v: [260.5, 259.5, 277.6, 377.7, 2249.6, 8369.0] },
      { name: 'fused (LL at small sizes)', color: C.fused, v: [81.5, 83.1, 91.5, 267.7, 2131.0, 8196.9] },
      { name: 'NCCL', color: C.nccl, v: [103.7, 112.5, 128.2, 321.1, 2329.3, 8383.6] }] },
    mp: { sizes: S10, series: [
      { name: 'direct read', color: C.direct, v: [15.8, 17.1, 18.9, 27.7, 59.3, 156.5, 558.4, 2162.5, 8581.8, 34170.5] },
      { name: 'fused ring (LL at small sizes)', color: C.fused, v: [42.2, 44.7, 45.2, 45.8, 56.7, 107.0, 228.8, 588.1, 2092.0, 8158.0] },
      { name: 'NCCL', color: C.nccl, v: [34.5, 38.0, 41.6, 41.9, 51.5, 103.4, 239.0, 630.6, 2271.8, 8291.2] },
      { name: 'NCCL, one process', color: C.nccl, dash: true, v: [103.7, 108.5, 112.5, 113.2, 128.2, 184.5, 321.1, 719.1, 2329.3, 8383.6] }] },
    // 8x H100, results/cuda_ring_8xH100_run2
    h100: { sizes: S10, series: [
      { name: 'fused ring (LL at small sizes)', color: C.fused, v: [70.6, 72.2, 76.1, 79.6, 92.2, 129.6, 200.0, 443.0, 1501.1, 5689.1] },
      { name: 'NCCL', color: C.nccl, v: [76.0, 77.0, 82.0, 83.2, 88.9, 110.8, 181.8, 381.4, 1155.4, 4071.4] },
      { name: 'NCCL, in-switch reduction off', color: C.nccl, dash: true, v: [77.1, 79.4, 81.7, 84.7, 89.5, 114.6, 182.9, 422.0, 1408.1, 5292.0] }] },
    h100mp: { sizes: S10, series: [
      { name: 'direct read', color: C.direct, v: [13.1, 13.4, 14.3, 18.9, 40.2, 108.3, 381.6, 1485.9, 5900.2, 23600.0] },
      { name: 'fused ring (LL at small sizes)', color: C.fused, v: [42.3, 45.0, 47.5, 49.6, 66.3, 105.1, 173.3, 415.3, 1476.2, 5660.2] },
      { name: 'NCCL', color: C.nccl, v: [27.2, 28.1, 29.5, 29.6, 35.4, 62.4, 134.2, 334.8, 1113.9, 4018.8] },
      { name: 'NCCL, in-switch reduction off', color: C.nccl, dash: true, v: [27.6, 27.6, 29.3, 30.2, 35.9, 59.0, 129.9, 375.5, 1363.3, 5226.7] }] },
    // 8x B200, results/cuda_ring_8xB200
    b200: { sizes: S10, series: [
      { name: 'fused ring (LL at small sizes)', color: C.fused, v: [79.1, 78.7, 80.6, 80.8, 88.0, 109.7, 331.2, 675.2, 2136.5, 8219.1] },
      { name: 'NCCL', color: C.nccl, v: [99.1, 95.5, 106.8, 96.5, 101.9, 122.9, 172.8, 350.7, 793.0, 2687.3] },
      { name: 'NCCL, in-switch reduction off', color: C.nccl, dash: true, v: [95.5, 101.0, 99.4, 95.0, 107.9, 118.8, 150.6, 332.9, 821.1, 2908.7] }] },
    b200mp: { sizes: S10, series: [
      { name: 'direct read', color: C.direct, v: [17.9, 19.4, 20.4, 22.6, 33.2, 66.6, 203.3, 746.5, 2902.5, 11568.5] },
      { name: 'fused ring (LL at small sizes)', color: C.fused, v: [44.3, 45.3, 46.7, 48.1, 52.9, 76.7, 298.2, 644.6, 2090.2, 8206.0] },
      { name: 'NCCL', color: C.nccl, v: [38.1, 39.0, 41.6, 41.1, 42.4, 67.2, 113.2, 295.1, 734.8, 2630.5] },
      { name: 'NCCL, in-switch reduction off', color: C.nccl, dash: true, v: [37.5, 38.7, 39.9, 41.2, 43.2, 56.0, 92.5, 266.5, 754.0, 2847.1] }] },
  };
  const MACHINES = [ // algorithm bandwidth at 256 MB per GPU
    { name: '2x T4, PCIe 3.0', bars: [['copy, async', 4.7, 'ring'], ['NCCL', 4.0, 'nccl']] },
    { name: '4x L4, PCIe 4.0, no peer access', bars: [['ring, copies', 6.4, 'ring'], ['NCCL', 3.7, 'nccl']] },
    { name: '8x A100, NVLink', bars: [['ring, copies', 90, 'ring'], ['ring, fused', 126, 'fused'], ['NCCL', 116, 'nccl']] },
    { name: '8x H100, NVLink', bars: [['ring, copies', 145, 'ring'], ['ring, fused', 186, 'fused'], ['NCCL', 232, 'nccl']] },
    { name: '8x B200, NVLink', bars: [['ring, copies', 226, 'ring'], ['ring, fused', 317, 'fused'], ['NCCL', 338, 'nccl']] },
  ];

  // ---------------------------------------------------------------- helpers
  const lx = Math.log10;
  const fmtB = b => b >= 1 << 30 ? (b / (1 << 30)) + ' GB' : b >= 1 << 20 ? (b / (1 << 20)) + ' MB' : (b / 1024) + ' KB';
  const fmtU = u => u >= 1000 ? (u / 1000).toFixed(u >= 10000 ? 0 : 1) + ' ms' : (u >= 100 ? Math.round(u) : u.toFixed(1)) + ' µs';
  const fmtG = g => g >= 10 ? Math.round(g) + ' GB/s' : g >= 1 ? g.toFixed(1) + ' GB/s' : g >= 0.01 ? g.toFixed(2) + ' GB/s' : g.toFixed(3) + ' GB/s';
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) {
      if (attrs[k] == null) continue;
      e.setAttribute(k, tag === 'rect' && k === 'rx' ? 0 : attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  const html = (tag, cls, parent, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  };
  const insertBeforeCaption = (fig, node) => fig.insertBefore(node, fig.querySelector('figcaption'));

  // ---------------------------------------------------------------- log-log chart; the post chooses one metric per experiment
  function chartSVG(ch, mode, W) {
    const narrow = W < 600, H = narrow ? 300 : 360, L = narrow ? 52 : 64, R = narrow ? 8 : 16, T = 14, B = 44;
    const sizes = ch.sizes, x0 = lx(sizes[0]), x1 = lx(sizes[sizes.length - 1]);
    // 'rel' plots speed relative to NCCL on a log2 axis, so bulk differences stay visible.
    const rel = mode === 'rel', ref = rel && ch.series.find(s => s.name === 'NCCL');
    const base = rel ? 2 : 10, ly = rel ? Math.log2 : lx;
    const yv = (s, i) => rel ? ref.v[i] / s.v[i] : s.v[i];
    let ymin = Infinity, ymax = -Infinity;
    for (const s of ch.series) s.v.forEach((u, i) => { const y = yv(s, i); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); });
    const y0 = rel ? Math.min(-1, Math.floor(ly(ymin) + 0.02)) : Math.floor(ly(ymin));
    const y1 = rel ? Math.max(1, Math.ceil(ly(ymax) - 0.02)) : Math.ceil(ly(ymax) + 0.05);
    const X = s => L + (lx(s) - x0) / (x1 - x0) * (W - L - R);
    const Y = y => T + (y1 - ly(y)) / (y1 - y0) * (H - T - B);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
    const grid = el('g', { class: 'grid' }, svg), axis = el('g', { class: 'axis' }, svg);
    el('path', { d: `M${L},${T} V${H - B} H${W - R}`, fill: 'none' }, axis);
    for (let d = y0; d <= y1; d++) {
      const t = Math.pow(base, d);
      const gl = el('line', { x1: L, x2: W - R, y1: Y(t), y2: Y(t) }, grid);
      if (rel && d === 0) gl.setAttribute('style', 'stroke:#aab0b4');
      const lab = rel ? t + 'x' : (t >= 1000 ? (t / 1000) + ' ms' : t + ' µs');
      el('text', { x: L - 8, y: Y(t) + 4, 'text-anchor': 'end' }, axis).textContent = lab;
    }
    (ch.ticks || sizes).forEach((s, i) => {
      if (narrow && i % 2) return;
      el('text', { x: X(s), y: H - B + 18, 'text-anchor': 'middle' }, axis).textContent = fmtB(s);
    });
    el('text', { x: L + (W - L - R) / 2, y: H - 4, 'text-anchor': 'middle' }, axis).textContent = 'message size per GPU';
    if (!narrow) el('text', { transform: `translate(12,${T + (H - T - B) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, axis)
      .textContent = rel ? 'speed relative to NCCL' : 'time per all-reduce';
    else el('text', { x: L - 8, y: T - 2, 'text-anchor': 'end' }, axis).textContent = rel ? 'vs NCCL' : '';
    for (const s of ch.series) {
      const d = s.v.map((u, i) => (i ? 'L' : 'M') + X(sizes[i]).toFixed(1) + ',' + Y(yv(s, i)).toFixed(1)).join(' ');
      const p = el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, svg);
      if (s.dash) { p.setAttribute('stroke-dasharray', '6 5'); p.setAttribute('opacity', '0.75'); continue; }
      if (ch.noDots) continue;
      if (rel && s === ref) continue;
      s.v.forEach((u, i) => el('circle', { cx: X(sizes[i]), cy: Y(yv(s, i)), r: 2.7, fill: '#fff', stroke: s.color, 'stroke-width': 1.4 }, svg));
    }
    const xh = el('line', { y1: T, y2: H - B, stroke: '#999', 'stroke-dasharray': '3 3', style: 'display:none' }, svg);
    const hit = el('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' }, svg);
    return { svg, X, Y, xh, hit, W, H, L, R, T, B };
  }

  function mountChart(fig) {
    const ch = CHARTS[fig.dataset.chart];
    const mode = fig.dataset.mode || 'lat';
    const legend = html('div', 'ar-legend');
    for (const s of ch.series) {
      const sp = html('span', '', legend);
      sp.style.color = s.color;
      const i = html('i', s.dash ? 'dash' : '', sp);
      if (!s.dash) i.style.background = s.color;
      html('span', '', sp, s.name).style.color = 'var(--ar-ink2)';
    }
    const plot = html('div');
    const tip = html('div', 'ar-tip');
    insertBeforeCaption(fig, legend); insertBeforeCaption(fig, plot); fig.appendChild(tip);
    let lastW = 0;
    function draw(force) {
      const W = Math.max(300, Math.round(plot.clientWidth || 800));
      if (!force && W === lastW) return;
      lastW = W;
      renderChart(fig, plot, tip, ch, mode, W, fig.querySelector('.ar-title').textContent + ', ' + (mode === 'rel' ? 'speed relative to NCCL' : 'latency') + ' against message size');
    }
    draw(true);
    if (window.ResizeObserver) new ResizeObserver(() => draw(false)).observe(plot);
  }

  function renderChart(fig, plot, tip, ch, mode, W, label) {
      const r = chartSVG(ch, mode, W);
      r.svg.setAttribute('aria-label', label);
      plot.replaceChildren(r.svg);
      const move = ev => {
        const rc = r.svg.getBoundingClientRect(), px = (ev.clientX - rc.left) / rc.width * W;
        let best = 0, bd = 1e9;
        ch.sizes.forEach((s, i) => { const d = Math.abs(r.X(s) - px); if (d < bd) { bd = d; best = i; } });
        r.xh.setAttribute('x1', r.X(ch.sizes[best])); r.xh.setAttribute('x2', r.X(ch.sizes[best])); r.xh.style.display = '';
        const size = ch.sizes[best];
        tip.innerHTML = '<div style="margin-bottom:3px"><b>' + fmtB(size) + '</b> per GPU</div>' + ch.series.map(s => {
          const u = s.v[best], g = size / u / 1000;
          const nc = ch.series.find(q => q.name === 'NCCL');
          const txt = mode === 'rel' ? (nc.v[best] / u).toFixed(2) + 'x (' + fmtG(g) + ', ' + fmtU(u) + ')'
            : fmtU(u) + ' (' + fmtG(g) + ')';
          return '<div><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:' + s.color + ';margin-right:6px"></span>' + s.name + ': <b>' + txt + '</b></div>';
        }).join('');
        tip.style.display = 'block';
        const fr = fig.getBoundingClientRect();
        let tx = ev.clientX - fr.left + 14;
        if (tx + tip.offsetWidth > fr.width - 8) tx = ev.clientX - fr.left - tip.offsetWidth - 14;
        tip.style.left = Math.max(4, tx) + 'px'; tip.style.top = (ev.clientY - fr.top + 12) + 'px';
      };
      const hide = () => { tip.style.display = 'none'; r.xh.style.display = 'none'; };
      r.hit.addEventListener('mousemove', move);
      r.hit.addEventListener('mouseleave', hide);
      r.hit.addEventListener('touchstart', e => move(e.touches[0]), { passive: true });
      r.hit.addEventListener('touchmove', e => move(e.touches[0]), { passive: true });
      r.hit.addEventListener('touchend', hide);
      return r;
  }

  // ---------------------------------------------------------------- bars per machine
  function mountBars(fig) {
    const legend = html('div', 'ar-legend');
    legend.innerHTML = [['ring, copies', 'ring'], ['ring, fused kernel', 'fused'], ['NCCL', 'nccl']]
      .map(([n, k]) => '<span><i style="background:var(--ar-' + k + ')"></i>' + n + '</span>').join('');
    const panels = html('div', 'ar-panels');
    for (const m of MACHINES) {
      const max = Math.max(...m.bars.map(b => b[1])) * 1.35;
      const best = m.bars.filter(b => b[2] !== 'nccl').reduce((a, b) => a[1] > b[1] ? a : b);
      const nccl = m.bars.find(b => b[2] === 'nccl');
      const p = html('div', 'ar-panel', panels);
      p.innerHTML = '<h5>' + m.name + '</h5>' + m.bars.map(b => {
        const w = (b[1] / max * 100).toFixed(1) + '%';
        return '<div class="ar-bar"><span class="lab">' + b[0] + '</span><div class="track" style="--w:' + w + '"><div class="fill" style="width:' + w +
          ';background:var(--ar-' + b[2] + ')"></div><span class="val">' + (b[1] < 10 ? b[1].toFixed(1) : b[1]) + '</span></div></div>';
      }).join('') + '<div class="ar-ratio">best hand-written / NCCL: ' + (best[1] / nccl[1]).toFixed(1) + 'x</div>';
    }
    insertBeforeCaption(fig, legend); insertBeforeCaption(fig, panels);
  }

  // ---------------------------------------------------------------- topology, with data flowing along each link
  // Rebuild a figure's SVG when the figure crosses the phone-width threshold.
  function responsive(fig, build) {
    let narrow = null;
    const check = () => {
      const n = (fig.clientWidth || 800) < 560;
      if (n !== narrow) { narrow = n; build(n); }
    };
    check();
    if (window.ResizeObserver) new ResizeObserver(check).observe(fig);
  }

  function mountTopology(fig) {
    let svg = null;
    responsive(fig, narrow => {
      const next = el('svg', { viewBox: narrow ? '0 0 430 330' : '0 0 860 300', role: 'img', 'aria-label': 'Paths between GPUs inside a machine and between machines' });
      const label = (x, y, s, opt) => {
        const t = el('text', { x, y, 'text-anchor': (opt && opt.anchor) || 'start', 'font-size': (opt && opt.size) || 13, fill: (opt && opt.color) || '#555',
          stroke: '#fff', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' }, next);
        t.textContent = s;
      };
      const box = (x, y, w, h, s) => {
        el('rect', { x, y, width: w, height: h, rx: 5, fill: '#f0f0f0', stroke: '#bbb' }, next);
        el('text', { x: x + w / 2, y: y + h / 2 + 5, 'text-anchor': 'middle', 'font-size': 14, fill: '#333' }, next).textContent = s;
      };
      const link = (d, color, width, slow) => {
        el('path', { d, fill: 'none', stroke: color, 'stroke-width': width, 'stroke-linecap': 'round' }, next);
      };
      const pcie = '#626b73', nv = '#218044', net = '#c65c12';
      el('rect', { x: 10, y: 10, width: 410, height: 280, rx: 8, fill: 'none', stroke: '#ccc' }, next);
      label(24, 34, 'Machine A', { color: '#333', size: 15 });
      link('M190,82 V110 H70 V140', pcie, 2, true); link('M190,110 H170 V140', pcie, 2, true);
      link('M190,110 H270 V140', pcie, 2, true); link('M190,110 H368 V140', pcie, 2, true);
      link('M230,65 H300', pcie, 2, true);
      [70, 170, 270, 368].forEach(x => link(`M${x},180 V236`, nv, 4));
      if (narrow) {
        link('M348,48 V22 H428', net, 3);
        label(424, 318, 'network to other machines: 12 to 50 GB/s per port', { anchor: 'end', color: net, size: 13 });
      } else {
        el('rect', { x: 560, y: 10, width: 290, height: 280, rx: 8, fill: 'none', stroke: '#ccc' }, next);
        label(574, 34, 'Machine B', { color: '#333', size: 15 });
        link('M765,82 V140', pcie, 2, true); link('M676,65 H725', pcie, 2, true);
        link('M705,180 V236', nv, 4);
        link('M396,65 H580', net, 3);
        box(580, 48, 96, 34, 'NIC'); box(725, 48, 80, 34, 'CPU');
        box(580, 140, 250, 40, 'GPUs 0 to 7'); box(580, 236, 250, 30, 'NVSwitch');
        label(488, 52, 'network', { anchor: 'middle', color: net });
        label(488, 90, '12 to 50 GB/s per port,', { anchor: 'middle', size: 12.5 });
        label(488, 106, 'shared by the machine', { anchor: 'middle', size: 12.5 });
      }
      box(150, 48, 80, 34, 'CPU'); box(300, 48, 96, 34, 'NIC');
      box(30, 140, 80, 40, 'GPU 0'); box(130, 140, 80, 40, 'GPU 1'); box(230, 140, 80, 40, 'GPU 2'); box(330, 140, 76, 40, 'GPU 3');
      box(30, 236, 376, 30, 'NVSwitch');
      label(200, 104, 'PCIe: 16 to 32 GB/s per card', { color: pcie, size: 12.5 });
      if (!narrow) el('rect', { x: 166, y: 199, width: 8, height: 18, fill: '#fff' }, next);
      label(narrow ? 215 : 218, 213, narrow ? 'NVLink: 300 to 900 GB/s per GPU' : 'NVLink: 300 to 900 GB/s per direction per GPU', { anchor: 'middle', color: nv, size: 12.5 });
      if (svg) svg.replaceWith(next); else insertBeforeCaption(fig, next);
      svg = next;
    });
  }

  // ---------------------------------------------------------------- animated ring all-reduce on 4 GPUs
  function mountRing(fig) {
    const P = 4, GC = ['#376789', '#ad6245', '#567b65', '#817194'], FULL = (1 << P) - 1;
    // states[k][gpu][chunk] = bitmask of GPUs whose values are summed into that chunk
    const states = [[...Array(P)].map((_, r) => Array(P).fill(1 << r))], moves = [];
    for (let ph = 0; ph < 2; ph++) for (let st = 0; st < P - 1; st++) {
      const old = states[states.length - 1], nw = old.map(row => row.slice()), mv = [];
      for (let r = 0; r < P; r++) {
        const dst = (r + 1) % P, c = ph === 0 ? (r - st + P) % P : (r + 1 - st + P) % P;
        nw[dst][c] = ph === 0 ? (old[dst][c] | old[r][c]) : old[r][c];
        mv.push({ src: r, dst, c, mask: old[r][c] });
      }
      states.push(nw); moves.push({ ph, mv });
    }
    const CW = 40, CH = 44;
    // wide: GPUs around a square, clockwise. narrow: GPUs stacked, wrap-around arrow on the left.
    const LAYOUT = {
      wide: { view: '0 0 760 296', BX: [40, 500, 500, 40], BY: [16, 16, 176, 176],
        arrows: ['M262,68 H496', 'M610,122 V172', 'M498,228 H264', 'M150,174 V124'], hint: [380, 153] },
      narrow: { view: '0 0 290 470', BX: [52, 52, 52, 52], BY: [6, 124, 242, 360],
        arrows: ['M162,110 V120', 'M162,228 V238', 'M162,346 V356', 'M50,412 H22 V58 H48'], hint: null },
    };
    const BW = 220, BH = 104;
    const controls = html('div', 'ar-controls');
    const bBack = html('button', '', controls, 'Previous'), bPlay = html('button', '', controls, ''), bNext = html('button', '', controls, 'Next');
    [bBack, bPlay, bNext].forEach(b => b.type = 'button');
    const text = html('div', 'ar-step');
    let svg = null, cells, arrows, packets, lay;
    const cellX = (g, c) => lay.BX[g] + 18 + c * 48, cellY = g => lay.BY[g] + 36;

    function paint(cell, mask, hl) {
      cell.bands.forEach((b, j) => b.setAttribute('fill', mask & (1 << j) ? GC[j] : '#fff'));
      cell.labels.forEach((t, j) => t.setAttribute('visibility', mask & (1 << j) ? 'visible' : 'hidden'));
      const full = mask === FULL;
      cell.frame.setAttribute('stroke', hl ? '#a65040' : full ? '#222' : '#aaa');
      cell.frame.setAttribute('stroke-width', hl || full ? 2.5 : 1);
    }
    function drawCell(grp, mask) {
      const bands = [], labels = [];
      el('rect', { x: 0, y: 0, width: CW, height: CH, fill: '#fff' }, grp);
      for (let j = 0; j < P; j++) {
        bands.push(el('rect', { x: 0, y: j * (CH / P), width: CW, height: CH / P, fill: '#fff', opacity: 0.15 }, grp));
        const t = el('text', { x: CW / 2, y: j * (CH / P) + 9, 'text-anchor': 'middle', 'font-size': 9, fill: GC[j] }, grp);
        t.textContent = 'GPU ' + j;
        labels.push(t);
      }
      const cell = { bands, labels, frame: el('rect', { x: 0, y: 0, width: CW, height: CH, fill: 'none', stroke: '#aaa', 'stroke-width': 1 }, grp) };
      paint(cell, mask);
      return cell;
    }
    function build(narrow) {
      lay = narrow ? LAYOUT.narrow : LAYOUT.wide;
      const next = el('svg', { viewBox: lay.view, role: 'img', 'aria-label': 'Animated ring all-reduce on four GPUs' });
      const mk = el('marker', { id: 'ar-ah', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, el('defs', {}, next));
      el('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#999' }, mk);
      arrows = lay.arrows.map(d => el('path', { d, fill: 'none', stroke: '#999', 'stroke-width': 2, 'marker-end': 'url(#ar-ah)' }, next));
      if (lay.hint) el('text', { x: lay.hint[0], y: lay.hint[1], 'text-anchor': 'middle', 'font-size': 14, fill: '#888' }, next).textContent = 'each GPU sends to its right neighbour';
      cells = [];
      for (let g = 0; g < P; g++) {
        el('rect', { x: lay.BX[g], y: lay.BY[g], width: BW, height: BH, rx: 6, fill: '#f6f6f6', stroke: '#bbb' }, next);
        el('rect', { x: lay.BX[g] + 12, y: lay.BY[g] + 11, width: 10, height: 10, rx: 2, fill: GC[g] }, next);
        el('text', { x: lay.BX[g] + 28, y: lay.BY[g] + 21, 'font-size': 14, fill: '#333' }, next).textContent = 'GPU ' + g;
        cells.push([]);
        for (let c = 0; c < P; c++) {
          cells[g].push(drawCell(el('g', { transform: `translate(${cellX(g, c)},${cellY(g)})` }, next), states[k][g][c]));
          el('text', { x: cellX(g, c) + CW / 2, y: cellY(g) + CH + 14, 'text-anchor': 'middle', 'font-size': 11, fill: '#888' }, next).textContent = 'chunk ' + c;
        }
      }
      packets = el('g', {}, next);
      if (svg) svg.replaceWith(next); else insertBeforeCaption(fig, next);
      svg = next;
    }

    const describe = n => {
      if (n === 0) return '<b>Start.</b> Each GPU holds its own vector, split into 4 chunks. Labels inside each chunk show which GPUs have contributed to its sum.';
      if (n <= P - 1) return '<b>Reduce-scatter, step ' + n + ' of ' + (P - 1) + '.</b> Every GPU sends one chunk to its right neighbour, which adds it to its own copy.' +
        (n === P - 1 ? ' Each GPU now owns one fully summed chunk (dark outline).' : '');
      const a = n - (P - 1);
      if (a < P - 1) return '<b>All-gather, step ' + a + ' of ' + (P - 1) + '.</b> Every GPU passes a finished chunk on to its right neighbour, which overwrites its copy.';
      return '<b>Done.</b> Every GPU has all four sums after 6 steps. Each GPU sent 6 chunks, 1.5 times its vector. In general that is 2(P-1)/P, which stays under 2 however many GPUs you add.';
    };
    let k = 0, playing = !reduceMotion, busy = false, timer = null, visible = false, gen = 0;
    function show(n) {
      k = n;
      for (let g = 0; g < P; g++) for (let c = 0; c < P; c++) paint(cells[g][c], states[k][g][c]);
      arrows.forEach(a => a.setAttribute('stroke', '#999'));
      text.innerHTML = describe(k);
      bBack.disabled = k === 0; bNext.disabled = k === states.length - 1;
    }
    function animate(n, done) { // move the packets for transition n -> n+1
      busy = true;
      const my = gen, m = moves[n];
      arrows.forEach(a => a.setAttribute('stroke', m.ph === 0 ? '#376789' : '#567b65'));
      const ps = m.mv.map(mv => {
        paint(cells[mv.src][mv.c], states[n][mv.src][mv.c], true);
        const grp = el('g', {}, packets);
        drawCell(grp, mv.mask);
        const x0 = cellX(mv.src, mv.c), y0 = cellY(mv.src), x1 = cellX(mv.dst, mv.c), y1 = cellY(mv.dst);
        const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1, bulge = lay === LAYOUT.narrow && mv.dst === 0 ? 70 : 26; // wrap-around goes out to the left
        return { grp, x0, y0, dx, dy, nx: dy / len * bulge, ny: -dx / len * bulge };
      });
      text.innerHTML = describe(n + 1);
      const dur = 1100, t0 = performance.now();
      const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      (function frame(now) {
        if (my !== gen) return; // layout was rebuilt mid-move
        const t = Math.min(1, (now - t0) / dur), e = ease(t), s = Math.sin(Math.PI * t);
        for (const p of ps) p.grp.setAttribute('transform', `translate(${p.x0 + p.dx * e + p.nx * s},${p.y0 + p.dy * e + p.ny * s})`);
        if (t < 1) return requestAnimationFrame(frame);
        packets.replaceChildren();
        busy = false;
        show(n + 1);
        if (done) done();
      })(t0);
    }
    function schedule() {
      clearTimeout(timer);
      if (!playing || !visible) return;
      const wait = k === 0 ? 1600 : k === states.length - 1 ? 3200 : 900;
      timer = setTimeout(() => {
        if (!playing || busy) return;
        if (k === states.length - 1) { show(0); schedule(); return; }
        animate(k, schedule);
      }, wait);
    }
    function setPlaying(p) { playing = p; bPlay.textContent = p ? 'Pause' : 'Play'; schedule(); }
    bPlay.addEventListener('click', () => setPlaying(!playing));
    bNext.addEventListener('click', () => {
      if (busy || k === states.length - 1) return;
      setPlaying(false);
      if (reduceMotion) show(k + 1); else animate(k);
    });
    bBack.addEventListener('click', () => { if (busy || k === 0) return; setPlaying(false); show(k - 1); });

    responsive(fig, narrow => {
      gen++;
      const wasBusy = busy;
      busy = false;
      build(narrow);
      if (wasBusy) k = Math.min(k + 1, states.length - 1);
      show(k);
      if (wasBusy) schedule();
    });
    insertBeforeCaption(fig, controls); insertBeforeCaption(fig, text);
    show(k); setPlaying(playing);
    if (window.IntersectionObserver) {
      new IntersectionObserver(es => { visible = es[0].isIntersecting; schedule(); }, { threshold: 0.4 }).observe(fig);
    } else { visible = true; schedule(); }
  }


  // Redraw a figure at its real pixel width, so text stays readable on phones.
  function atWidth(node, draw) {
    let lastW = 0;
    const go = force => {
      const W = Math.max(300, Math.round(node.clientWidth || 760));
      if (force || W !== lastW) { lastW = W; draw(W); }
    };
    go(true);
    if (window.ResizeObserver) new ResizeObserver(() => go(false)).observe(node);
    return () => go(true);
  }
  const fmtS = b => b >= 1 << 30 ? (b / (1 << 30)).toFixed(1) + ' GB' : b >= 1 << 20 ? (b / (1 << 20)).toFixed(b >= 10 << 20 ? 0 : 1) + ' MB' : Math.round(b / 1024) + ' KB';

  // ---------------------------------------------------------------- cost model: steps x alpha + bytes / B
  function mountModel(fig) {
    const LINKS = { pcie: { name: 'PCIe', a: 20, B: 10 }, nvlink: { name: 'NVLink', a: 5, B: 200 } }; // illustrative, us and GB/s
    let P = 8, link = 'nvlink';
    const head = fig.querySelector('.ar-head');
    const tools = html('div', 'ar-tools', head);
    const tP = html('div', 'ar-toggle', tools), tL = html('div', 'ar-toggle', tools);
    tP.innerHTML = [2, 4, 8].map(p => '<button type="button" data-p="' + p + '">' + p + ' GPUs</button>').join('');
    tL.innerHTML = Object.keys(LINKS).map(k => '<button type="button" data-l="' + k + '">' + LINKS[k].name + '</button>').join('');
    const legend = html('div', 'ar-legend');
    legend.innerHTML = '<span><i style="background:var(--ar-ring)"></i>ring</span><span><i style="background:var(--ar-direct)"></i>direct read</span>';
    const plot = html('div'), tip = html('div', 'ar-tip'), text = html('div', 'ar-step');
    insertBeforeCaption(fig, legend); insertBeforeCaption(fig, plot); insertBeforeCaption(fig, text); fig.appendChild(tip);
    const sizes = []; for (let e = 12; e <= 30.001; e += 0.375) sizes.push(Math.pow(2, e));
    const redraw = atWidth(plot, W => {
      const { a, B } = LINKS[link];
      tP.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.p === P));
      tL.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.l === link));
      const ring = sizes.map(S => 2 * (P - 1) * a + 2 * (P - 1) / P * S / (B * 1e3));
      const direct = sizes.map(S => a + (P - 1) * S / (B * 1e3));
      const ch = { sizes, ticks: S10, noDots: true, series: [
        { name: 'ring', color: C.ring, v: ring }, { name: 'direct read', color: C.direct, v: direct }] };
      const r = renderChart(fig, plot, tip, ch, 'lat', W, 'Modelled all-reduce time against message size, ring and direct read');
      const svg = r.svg, top = r.T, bot = r.H - r.B;
      const note = (x, y, s, anchor, color) => {
        const t = el('text', { x, y, 'text-anchor': anchor, 'font-size': 12, fill: color || 'var(--ar-ink2)', stroke: '#fff', 'stroke-width': 4, 'paint-order': 'stroke', 'pointer-events': 'none' }, svg);
        t.textContent = s;
      };
      const ddp = 25 * (1 << 20), xd = r.X(ddp);
      el('line', { x1: xd, x2: xd, y1: bot - 10, y2: bot, stroke: 'var(--ar-ink2)', 'stroke-width': 2, 'pointer-events': 'none' }, svg);
      note(xd, bot - 16, 'DDP bucket, 25 MiB', 'middle');
      let msg;
      if (P === 2) {
        msg = '<b>Two GPUs:</b> both algorithms move $S$ bytes per GPU, so the ring’s extra step never pays for itself and direct read is ahead at every size.';
      } else {
        const Sx = a * (2 * P - 3) * P * B * 1e3 / ((P - 1) * (P - 2));
        const x = r.X(Sx);
        el('line', { x1: x, x2: x, y1: top, y2: bot, stroke: '#999', 'stroke-dasharray': '4 4', 'pointer-events': 'none' }, svg);
        note(x - 8, top + 14, '← direct faster', 'end', C.direct);
        note(x + 8, top + 14, 'ring faster →', 'start', C.ring);
        msg = '<b>Crossover at about ' + fmtS(Sx) + '.</b> Below it the ring’s ' + 2 * (P - 1) + ' steps cost more than direct read’s extra bytes. Above it direct read moves ' +
          ((P - 1) / (2 * (P - 1) / P)).toFixed(1) + 'x the ring’s traffic and falls behind.';
      }
      text.innerHTML = msg.replace('$S$', '<i>S</i>') + ' <span style="color:var(--ar-muted)">Parameters: α = ' + a + ' µs per step, B = ' + B + ' GB/s.</span>';
    });
    tP.addEventListener('click', e => { const b = e.target.closest('button'); if (b) { P = +b.dataset.p; redraw(); } });
    tL.addEventListener('click', e => { const b = e.target.closest('button'); if (b) { link = b.dataset.l; redraw(); } });
  }

  // ---------------------------------------------------------------- what all-reduce does: inputs -> the same sum on every GPU
  function mountAllReduce(fig) {
    const GC = ['#376789', '#ad6245', '#567b65', '#817194'];
    const IN = [[1, 2, 3, 0], [3, 4, 0, 1], [0, 1, 2, 4], [0, 1, 1, 5]];   // sums 4, 8, 6, 10: all different
    const P = IN.length, N = IN[0].length;
    const SUM = IN[0].map((_, i) => IN.reduce((a, row) => a + row[i], 0));
    const wrap = html('div'), text = html('div', 'ar-step');
    insertBeforeCaption(fig, wrap); insertBeforeCaption(fig, text);
    let hl = -1, cols = [];
    const describe = () => hl < 0
      ? 'Hover or tap a position. Every GPU ends up with the same vector, the elementwise sum of all four inputs.'
      : '<b>Position ' + hl + ':</b> ' + IN.map(r => r[hl]).join(' + ') + ' = ' + SUM[hl] + ', and every GPU ends up holding ' + SUM[hl] + ' there.';
    const paint = () => { cols.forEach((els, i) => els.forEach(e => e.setAttribute('opacity', hl < 0 || hl === i ? 1 : 0.2))); text.innerHTML = describe(); };
    atWidth(wrap, W => {
      const narrow = W < 560, M = 4, arrowW = narrow ? 0 : 110, arrowH = narrow ? 48 : 0;
      const blockW = narrow ? W - 2 * M : (W - 2 * M - arrowW) / 2;
      const labW = 54, gap = 6, rowH = 30, rowGap = 8, head = 24;
      const cw = (blockW - labW - (N - 1) * gap) / N;
      const blockH = head + P * rowH + (P - 1) * rowGap;
      const H = narrow ? 2 * blockH + arrowH + 6 : blockH + 6;
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Four GPUs each hold a vector. After all-reduce each holds the elementwise sum' });
      cols = [...Array(N)].map(() => []);
      const block = (x0, y0, title, after) => {
        el('text', { x: x0 + labW, y: y0 + 15, 'font-size': 13, fill: '#555' }, svg).textContent = title;
        for (let g = 0; g < P; g++) {
          const y = y0 + head + g * (rowH + rowGap);
          el('text', { x: x0, y: y + rowH / 2 + 5, 'font-size': 13, fill: '#333' }, svg).textContent = 'GPU ' + g;
          for (let i = 0; i < N; i++) {
            const x = x0 + labW + i * (cw + gap), grp = el('g', { style: 'cursor:pointer' }, svg);
            if (after) {
              el('rect', { x, y, width: cw, height: rowH, rx: 3, fill: '#fff', stroke: '#aab2b8' }, grp);
              el('text', { x: x + cw / 2, y: y + rowH / 2 + 3, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 'bold', fill: '#333' }, grp).textContent = SUM[i];
            } else {
              el('rect', { x, y, width: cw, height: rowH, rx: 0, fill: GC[g] + '18', stroke: GC[g], 'stroke-width': 1.5 }, grp);
              el('text', { x: x + cw / 2, y: y + rowH / 2 + 5, 'text-anchor': 'middle', 'font-size': 14, fill: '#333' }, grp).textContent = IN[g][i];
            }
            grp.addEventListener('mouseenter', () => { hl = i; paint(); });
            grp.addEventListener('click', () => { hl = hl === i ? -1 : i; paint(); });
            cols[i].push(grp);
          }
        }
      };
      block(M, 0, 'before: each GPU’s own vector', false);
      if (narrow) {
        const xm = M + labW + (blockW - labW) / 2, y0 = blockH + 6;
        el('path', { d: `M${xm},${y0} V${y0 + arrowH - 10}`, stroke: '#888', 'stroke-width': 2, fill: 'none' }, svg);
        el('path', { d: `M${xm - 6},${y0 + arrowH - 16} L${xm},${y0 + arrowH - 8} L${xm + 6},${y0 + arrowH - 16}`, stroke: '#888', 'stroke-width': 2, fill: 'none' }, svg);
        el('text', { x: xm + 12, y: y0 + arrowH / 2, 'font-size': 13, fill: '#555' }, svg).textContent = 'all-reduce (sum)';
        block(M, blockH + arrowH, 'after: the same sum on every GPU', true);
      } else {
        const xa = M + blockW + 14, xb = M + blockW + arrowW - 14, ym = head + (blockH - head) / 2;
        el('path', { d: `M${xa},${ym} H${xb}`, stroke: '#888', 'stroke-width': 2, fill: 'none' }, svg);
        el('path', { d: `M${xb - 8},${ym - 6} L${xb},${ym} L${xb - 8},${ym + 6}`, stroke: '#888', 'stroke-width': 2, fill: 'none' }, svg);
        el('text', { x: (xa + xb) / 2, y: ym - 10, 'text-anchor': 'middle', 'font-size': 13, fill: '#555' }, svg).textContent = 'all-reduce';
        el('text', { x: (xa + xb) / 2, y: ym + 22, 'text-anchor': 'middle', 'font-size': 12.5, fill: '#888' }, svg).textContent = '(sum)';
        block(M + blockW + arrowW, 0, 'after: the same sum on every GPU', true);
      }
      svg.addEventListener('mouseleave', () => { hl = -1; paint(); });
      wrap.replaceChildren(svg);
      paint();
    });
  }

  // ---------------------------------------------------------------- direct read: every GPU reads all inputs and sums them itself
  function mountDirect(fig) {
    const P = 4, GC = ['#376789', '#ad6245', '#567b65', '#817194'];
    const wrap = html('div'), text = html('div', 'ar-step');
    insertBeforeCaption(fig, wrap); insertBeforeCaption(fig, text);
    let sel = 0, cols = [], reads = [];
    const describe = () => '<b>GPU ' + sel + '</b> reads the other three inputs over NVLink (coloured) and its own locally (grey), ' +
      'and writes the sum to its own output. All four GPUs do this at the same time, in a single step, each reading 3 × S bytes from the other GPUs.';
    const paint = () => {
      reads.forEach((r, s) => r.setAttribute('display', s === sel ? 'inline' : 'none'));
      cols.forEach((c, g) => c.setAttribute('stroke', g === sel ? '#222' : '#bbb'));
      text.innerHTML = describe();
    };
    atWidth(wrap, W => {
      const narrow = W < 560, M = 4, gap = narrow ? 12 : 28;
      const bw = (W - 2 * M - (P - 1) * gap) / P, pad = narrow ? 6 : 12;
      const inY = 30, cellH = 30, outY = 150, H = outY + cellH + 14;
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Direct read: each GPU reads every input and sums them into its own output' });
      const defs = el('defs', {}, svg);
      [...GC, '#999'].forEach((c, k) => {
        const mk = el('marker', { id: 'ar-dr' + k, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, defs);
        el('path', { d: 'M0,0 L10,5 L0,10 z', fill: c }, mk);
      });
      const bx = g => M + g * (bw + gap), cx = g => bx(g) + bw / 2;
      cols = []; reads = [];
      for (let g = 0; g < P; g++) {
        cols.push(el('rect', { x: bx(g), y: 0, width: bw, height: H - 4, rx: 6, fill: '#f6f6f6', stroke: '#bbb' }, svg));
        el('text', { x: cx(g), y: 19, 'text-anchor': 'middle', 'font-size': 13.5, fill: '#333' }, svg).textContent = 'GPU ' + g;
      }
      for (let s = 0; s < P; s++) {   // the reads made by GPU s, shown when s is selected
        const grp = el('g', {}, svg);
        for (let g = 0; g < P; g++) {
          const local = g === s, col = local ? '#999' : GC[g];
          const x0 = cx(g) + (local ? 0 : (s - g) * 4), x1 = cx(s) + (g - 1.5) * (narrow ? 11 : 22);   // spread the four arrowheads along the output
          el('path', { d: `M${x0},${inY + cellH} C${x0},${inY + cellH + 50} ${x1},${outY - 50} ${x1},${outY - 2}`, fill: 'none', stroke: col,
            'stroke-width': local ? 2 : 2.5, 'stroke-dasharray': local ? '4 3' : null, 'marker-end': `url(#ar-dr${local ? 4 : g})` }, grp);
        }
        reads.push(grp);
      }
      for (let g = 0; g < P; g++) {
        el('rect', { x: bx(g) + pad, y: inY, width: bw - 2 * pad, height: cellH, rx: 0, fill: '#fff', stroke: GC[g] }, svg);
        el('text', { x: cx(g), y: inY + 20, 'text-anchor': 'middle', 'font-size': 13, fill: '#333' }, svg).textContent = narrow ? 'in' : 'input';
        el('rect', { x: bx(g) + pad, y: outY, width: bw - 2 * pad, height: cellH, rx: 3, fill: '#f0f0f0', stroke: '#bbb' }, svg);
        el('text', { x: cx(g), y: outY + 18, 'text-anchor': 'middle', 'font-size': 13, fill: '#333' }, svg).textContent = narrow ? 'sum' : 'sum of all 4';
        const hit = el('rect', { x: bx(g), y: 0, width: bw, height: H, fill: 'transparent', style: 'cursor:pointer' }, svg);
        hit.addEventListener('mouseenter', () => { sel = g; paint(); });
        hit.addEventListener('click', () => { sel = g; paint(); });
      }
      wrap.replaceChildren(svg);
      paint();
    });
  }

  // ---------------------------------------------------------------- who moves the bytes: copy engine vs SM loads, step by step
  function mountMovers(fig) {
    const REQ = '#376789', DATA = '#567b65';
    const MODES = {
      ce: { label: 'copy engine', used: ['host', 'ce0', 'mem0', 'sm0', 'mem1'], steps: [
        { hl: ['host', 'ce0'], arrows: [['host', 'ce0', REQ, false]],
          t: '<b>The host queues a copy.</b> <code>cudaMemcpyPeerAsync</code> submits the source address on GPU 1, destination in GPU 0’s memory and size to a stream.' },
        { hl: ['ce0', 'link', 'mem1'], arrows: [['ce0', 'mem1', REQ, true]],
          t: '<b>The copy engine issues read requests.</b> In this pull example, GPU 0’s DMA engine requests data from GPU 1 over the connection. GPU 0’s SMs are available for other work until the add kernel starts.' },
        { hl: ['mem1', 'link', 'ce0'], arrows: [['mem1', 'ce0', DATA, false]],
          t: '<b>GPU 1’s memory answers.</b> Its memory controller returns the data. GPU 1’s SMs and copy engines are not involved. It just serves memory.' },
        { hl: ['ce0', 'mem0'], arrows: [['ce0', 'mem0', DATA, false]],
          t: '<b>The data lands in a buffer.</b> The copy engine writes it into a temporary buffer in GPU 0’s memory. It can only copy, not add.' },
        { hl: ['sm0', 'mem0'], arrows: [['mem0', 'sm0', DATA, false], ['sm0', 'mem0', DATA, false]],
          t: '<b>Only now can the add start.</b> A separate kernel on GPU 0’s SMs reads both inputs from local memory, adds them and writes the result: a second pass over memory.' }] },
      sm: { label: 'SM loads', used: ['host', 'sm0', 'mem0', 'mem1'], steps: [
        { hl: ['host', 'sm0'], arrows: [['host', 'sm0', REQ, false]],
          t: '<b>The host launches one kernel.</b> Nothing else is submitted per transfer: the kernel’s threads will fetch the remote data themselves.' },
        { hl: ['sm0', 'link', 'mem1'], arrows: [['sm0', 'mem1', REQ, true]],
          t: '<b>Threads load from a remote address.</b> Their pointer points into GPU 1’s memory. The memory system merges a warp’s loads into larger requests and sends them across the link.' },
        { hl: ['sm0'], arrows: [['sm0', 'mem1', REQ, true]],
          t: '<b>The value is needed before the add.</b> A thread cannot use the loaded value until it arrives, though independent instructions and other warps can run meanwhile. Keeping enough loads outstanding helps cover that latency.' },
        { hl: ['mem1', 'link', 'sm0'], arrows: [['mem1', 'sm0', DATA, false]],
          t: '<b>GPU 1’s memory answers.</b> As before, GPU 1 just serves memory. The data goes straight into the waiting threads’ registers, not into a buffer.' },
        { hl: ['sm0', 'mem0'], arrows: [['mem0', 'sm0', DATA, false], ['sm0', 'mem0', DATA, false]],
          t: '<b>Add and store in the same pass.</b> Each thread adds the remote value to its local one and writes the result. Moving and adding happen together, with no temporary buffer.' }] },
    };
    // One compact layout per panel: GPU 0 left, GPU 1 right, link in the gap between them.
    // Arrows run at right angles through the gap row (GY) so they never pass under another box.
    const W = 380, H = 248, GY = 150;
    const B = {
      host: [100, 6, 180, 28, 'host (CPU)'], g0: [4, 50, 164, 176, 'GPU 0'], g1: [212, 50, 164, 176, 'GPU 1'],   // GPU names sit under their boxes
      sm0: [14, 82, 68, 48, 'SMs'], ce0: [90, 82, 68, 48, 'copy|engine'], mem0: [14, 170, 144, 44, 'memory'],
      ce1: [222, 82, 68, 48, 'copy|engine'], sm1: [298, 82, 68, 48, 'SMs'], mem1: [222, 170, 144, 44, 'memory'],
      link: [172, 134, 36, 32, 'link'] };
    const cx = b => b[0] + b[2] / 2, cy = b => b[1] + b[3] / 2;
    const near = (b, y) => y > cy(b) ? b[1] + b[3] + 3 : b[1] - 3;   // the box edge facing height y
    function route(from, to, off) {
      const a = B[from], z = B[to];
      if (from === 'host') {
        const x = cx(z);
        return x >= a[0] && x <= a[0] + a[2] ? [[x, a[1] + a[3] + 3], [x, z[1] - 3]]
          : [[x < a[0] ? a[0] - 3 : a[0] + a[2] + 3, cy(a)], [x, cy(a)], [x, z[1] - 3]];
      }
      if (from.slice(-1) !== to.slice(-1)) // across GPUs: down into the gap row, through the link, then up or down
        return [[cx(a), near(a, GY)], [cx(a), GY], [cx(z), GY], [cx(z), near(z, GY)]];
      const x = (from.startsWith('mem') ? cx(z) : cx(a)) + off;   // within a GPU: straight between unit and memory
      return [[x, near(a, cy(z))], [x, near(z, cy(a))]];
    }
    let k = 0;
    const N = Math.max(...Object.values(MODES).map(M => M.steps.length));
    const grid = html('div', 'ar-movers');
    insertBeforeCaption(fig, grid);
    const panels = Object.keys(MODES).map(m => {
      const p = html('div', 'ar-panel', grid);
      html('h5', '', p, 'With ' + MODES[m].label);
      const slot = html('div', '', p), text = html('div', 'ar-step', p);
      return { M: MODES[m], id: m, slot, text };
    });
    const legend = html('div', 'ar-legend');
    legend.innerHTML = `<span><i class="dash" style="color:${REQ}"></i>requests</span><span><i style="background:${DATA}"></i>data</span>`;
    const controls = html('div', 'ar-controls');
    const bBack = html('button', '', controls, 'Previous'), bNext = html('button', '', controls, 'Next'), cnt = html('span', '', controls);
    [bBack, bNext].forEach(b => b.type = 'button');
    cnt.style.cssText = 'font-size:0.8em;color:var(--ar-muted)';
    insertBeforeCaption(fig, legend); insertBeforeCaption(fig, controls);
    bBack.onclick = () => { if (k > 0) { k--; draw(); } };
    bNext.onclick = () => { if (k < N - 1) { k++; draw(); } };
    function panelSVG(P) {
      const M = P.M, st = M.steps[Math.min(k, M.steps.length - 1)];
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'One transfer from GPU 1 to GPU 0, by ' + M.label });
      const defs = el('defs', {}, svg);
      [[REQ, 'q'], [DATA, 'd']].forEach(([c, s]) => {
        const mk = el('marker', { id: `ar-mv${s}-${P.id}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto' }, defs);
        el('path', { d: 'M0,0 L10,5 L0,10 z', fill: c }, mk);
      });
      ['g0', 'g1'].forEach(g => {
        const b = B[g];
        el('rect', { x: b[0], y: b[1], width: b[2], height: b[3], rx: 8, fill: 'none', stroke: '#ccc' }, svg);
        el('text', { x: cx(b), y: b[1] + b[3] + 18, 'text-anchor': 'middle', 'font-size': 13, fill: '#333' }, svg).textContent = b[4];
      });
      st.arrows.forEach(([from, to, col], ai) => {   // drawn before the boxes, so the link box covers the part inside it
        const off = st.arrows.length > 1 ? (ai ? 14 : -14) : 0;
        el('path', { d: 'M' + route(from, to, off).map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' L'), fill: 'none', stroke: col,
          'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-dasharray': col === REQ ? '6 4' : null,
          'marker-end': `url(#ar-mv${col === REQ ? 'q' : 'd'}-${P.id})` }, svg);
      });
      for (const id of ['host', 'sm0', 'ce0', 'mem0', 'ce1', 'sm1', 'mem1', 'link']) {
        const b = B[id], on = st.hl.includes(id), used = M.used.includes(id) || id === 'link';
        el('rect', { x: b[0], y: b[1], width: b[2], height: b[3], rx: 5, fill: on ? '#edf2f5' : used ? '#f0f0f0' : '#fafafa',
          stroke: on ? '#376789' : used ? '#bbb' : '#ddd', 'stroke-width': on ? 2.5 : 1 }, svg);
        const lines = b[4].split('|').concat(used ? [] : ['(idle)']), lh = 14;
        lines.forEach((ln, i) => {
          el('text', { x: cx(b), y: cy(b) + 4.5 + (i - (lines.length - 1) / 2) * lh, 'text-anchor': 'middle',
            'font-size': id === 'link' ? 12 : i === lines.length - 1 && !used ? 11.5 : 13, fill: used ? '#333' : '#bbb' }, svg).textContent = ln;
        });
      }
      return svg;
    }
    function draw() {
      panels.forEach(P => {
        P.slot.replaceChildren(panelSVG(P));
        P.text.innerHTML = k < P.M.steps.length ? P.M.steps[k].t : '';
      });
      cnt.textContent = 'step ' + (k + 1) + ' of ' + N;
      bBack.disabled = k === 0; bNext.disabled = k === N - 1;
    }
    draw();
  }

  // ---------------------------------------------------------------- measured transfer rates against advertised link bandwidth
  function mountMoverLimit(fig) {
    const MACH = [
      { name: 'PCIe 3.0, two T4s', note: 'Both measured rates are below the advertised link bandwidth. This alone does not identify the bottleneck.', bars: [
        ['link rating', 16, '#ced4da'], ['copy engine', 9.9, 'var(--ar-copy)'], ['SM loads', 7.05, 'var(--ar-rr)']] },
      { name: 'NVLink 3, two A100s', note: 'Remote loads achieve more bandwidth in this comparison, but the copy engine’s intrinsic limit was not established.', bars: [
        ['link rating', 300, '#ced4da'], ['copy engine', 164, 'var(--ar-copy)'], ['SM loads', 233, 'var(--ar-rr)']] },
    ];
    const panels = html('div', 'ar-panels');
    for (const m of MACH) {
      const max = Math.max(...m.bars.map(b => b[1])) * 1.3, p = html('div', 'ar-panel', panels);
      p.innerHTML = '<h5>' + m.name + ', GB/s</h5>' + m.bars.map(b => {
        const w = (b[1] / max * 100).toFixed(1) + '%';
        return '<div class="ar-bar"><span class="lab">' + b[0] + '</span><div class="track" style="--w:' + w + '"><div class="fill" style="width:' + w +
          ';background:' + b[2] + '"></div><span class="val">' + (b[1] < 10 ? b[1].toFixed(1) : b[1]) + '</span></div></div>';
      }).join('') + '<div class="ar-ratio">' + m.note + '</div>';
    }
    insertBeforeCaption(fig, panels);
  }

  // ---------------------------------------------------------------- protocols: where the readiness flag lives
  function mountProtocols(fig) {
    const ROWS = [
      { name: 'Simple', pay: '~100%', note: 'The data is written first, then a separate flag. The receiver waits for the flag before reading.',
        cells: Array(16).fill('d'), flagApart: true },
      { name: 'LL', pay: '50%', note: 'Each 4-byte value is paired with a 4-byte flag. The receiver polls the data itself.', cells: Array(16).fill('h') },
      { name: 'LL128', pay: '94%', note: 'Each 128-byte line carries one 8-byte flag. Only safe where 128-byte stores arrive whole.',
        cells: Array(15).fill('d').concat(['f']) },
    ];
    const wrap = html('div');
    insertBeforeCaption(fig, wrap);
    atWidth(wrap, W => {
      const narrow = W < 560, nameW = narrow ? 0 : 74, payW = narrow ? 0 : 56;
      const stripW = W - nameW - payW - 8, cw = stripW / 18, ch = narrow ? 22 : 26;
      const rowH = narrow ? 86 : 80, top = 30, H = top + ROWS.length * rowH;
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'How the Simple, LL and LL128 protocols place readiness flags' });
      const D = 'var(--ar-ring)', F = 'var(--ar-nccl)';
      const key = (x, col, s) => { el('rect', { x, y: 6, width: 12, height: 12, rx: 2, fill: col }, svg); el('text', { x: x + 18, y: 16.5, 'font-size': 12.5, fill: '#555' }, svg).textContent = s; };
      key(nameW, D, 'data'); key(nameW + 70, F, 'flag');
      el('text', { x: nameW + 140, y: 16.5, 'font-size': 12, fill: '#888' }, svg).textContent = narrow ? '8 bytes per cell' : 'each cell is 8 bytes, 128 bytes shown';
      ROWS.forEach((row, i) => {
        const y0 = top + i * rowH + (narrow ? 20 : 4);
        const label = el('text', { x: narrow ? 0 : 0, y: narrow ? y0 - 6 : y0 + ch / 2 + 5, 'font-size': 14, fill: '#333', 'font-weight': 'bold' }, svg);
        label.textContent = row.name;
        const pay = el('text', { x: W, y: narrow ? y0 - 6 : y0 + ch / 2 + 5, 'text-anchor': 'end', 'font-size': 13, fill: '#555' }, svg);
        pay.textContent = row.pay + (narrow ? ' data' : '');
        row.cells.forEach((k, j) => {
          const x = nameW + j * cw;
          if (k === 'h') {
            el('rect', { x: x + 0.5, y: y0, width: cw / 2 - 0.5, height: ch, fill: D }, svg);
            el('rect', { x: x + cw / 2, y: y0, width: cw / 2 - 1, height: ch, fill: F }, svg);
          } else el('rect', { x: x + 0.5, y: y0, width: cw - 1.5, height: ch, rx: 1.5, fill: k === 'f' ? F : D }, svg);
        });
        if (row.flagApart) {
          el('text', { x: nameW + 16 * cw + cw / 2, y: y0 + ch / 2 + 5, 'text-anchor': 'middle', 'font-size': 13, fill: '#888' }, svg).textContent = '…';
          el('rect', { x: nameW + 17 * cw + 0.5, y: y0, width: cw - 1.5, height: ch, rx: 1.5, fill: F }, svg);
        }
        row.note.split(/(?<=\.) /).forEach((line, n) =>
          el('text', { x: nameW, y: y0 + ch + 17 + n * 15, 'font-size': 12.5, fill: '#666' }, svg).textContent = line);
      });
      if (!narrow) el('text', { x: W, y: 16.5, 'text-anchor': 'end', 'font-size': 12, fill: '#888' }, svg).textContent = 'data share';
      wrap.replaceChildren(svg);
    });
  }


  // ---------------------------------------------------------------- who is busy when: host lane and GPU lanes through section 6
  function mountTimeline(fig) {
    const P = 8, STEPS = 2 * (P - 1);
    const STAGES = {
      host: { name: 'Host-driven ring', color: C.ring, total: 1149,
        text: '<b>Host-driven ring.</b> One CPU thread issues all 112 copies, 14 per GPU, at about 10 µs each. Each GPU waits for its next copy to be issued. 1149 µs at 4 KB.' },
      fused: { name: 'Fused kernel', color: C.fused, total: 260.5,
        text: '<b>Fused kernel.</b> The host launches one kernel per GPU, and each kernel runs all 14 steps itself. Between steps, a GPU waits for its neighbour’s flag. 260.5 µs at 4 KB (the per-thread-fence tuning run).' },
      direct: { name: 'Direct read', color: C.direct, total: 45,
        text: '<b>Direct read.</b> One launch per GPU and one step: each kernel reads every other GPU’s input and sums. Four times the ring’s bytes, but no rounds between GPUs. Measured: 45 µs at 4 KB, with a single host process. The barrier that makes the inputs safe to read is outside the timing.' },
    };
    // data-stages="host,fused" limits the figure to those stages; default shows all.
    if (fig.dataset.stages) { const keep = fig.dataset.stages.split(','); Object.keys(STAGES).forEach(k => { if (!keep.includes(k)) delete STAGES[k]; }); }
    const sim = key => {
      const ev = [], end = Array.from({ length: P }, () => []);
      let T = 0;
      const block = (lane, t0, t1, kind, title) => { ev.push({ lane, t0, t1, kind, title }); T = Math.max(T, t1); };
      if (key === 'host') {
        const ti = 10, tw = 3;
        for (let st = 0; st < STEPS; st++) for (let r = 0; r < P; r++) {
          const k = st * P + r, src = (r + P - 1) % P;
          block(0, k * ti, (k + 1) * ti, 'issue', 'host issues copy ' + (k + 1) + ' of ' + P * STEPS + ' (GPU ' + r + ', step ' + (st + 1) + ')');
          const t0 = Math.max((k + 1) * ti, st ? Math.max(end[r][st - 1], end[src][st - 1]) : 0);
          end[r][st] = t0 + tw;
          block(1 + r, t0, t0 + tw, 'step', 'GPU ' + r + ', step ' + (st + 1) + ': copy a chunk from GPU ' + src + ', then add');
        }
      } else {
        const tl = 5;
        for (let r = 0; r < P; r++) block(0, r * tl, (r + 1) * tl, 'issue', 'host launches the kernel on GPU ' + r);
        if (key === 'fused') {
          const ts = 15;
          for (let st = 0; st < STEPS; st++) for (let r = 0; r < P; r++) {
            const src = (r + P - 1) % P;
            const t0 = Math.max(st ? end[r][st - 1] : (r + 1) * tl, st ? end[src][st - 1] : 0);
            end[r][st] = t0 + ts;
            block(1 + r, t0, t0 + ts, 'step', 'GPU ' + r + ', step ' + (st + 1) + ': read a chunk from GPU ' + src + ', add, publish a flag');
          }
        } else {
          for (let r = 0; r < P; r++) block(1 + r, (r + 1) * tl, (r + 1) * tl + 8, 'step', 'GPU ' + r + ': read all ' + (P - 1) + ' other inputs and sum');
        }
      }
      return { ev, T };
    };
    let stage = 'host';
    const head = fig.querySelector('.ar-head');
    const tools = html('div', 'ar-tools', head), tog = html('div', 'ar-toggle', tools);
    tog.innerHTML = Object.keys(STAGES).map(k => '<button type="button" data-s="' + k + '">' + STAGES[k].name + '</button>').join('');
    const wrap = html('div'), text = html('div', 'ar-step');
    insertBeforeCaption(fig, wrap); insertBeforeCaption(fig, text);
    const redraw = atWidth(wrap, W => {
      tog.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.s === stage));
      const S = STAGES[stage], { ev, T } = sim(stage);
      const narrow = W < 560, labelW = narrow ? 46 : 58, laneH = narrow ? 15 : 19, gap = 5, top = 20;
      const lanes = 1 + P, lanesH = lanes * (laneH + gap);
      const totTop = top + lanesH + 22, barH = 11, H = totTop + Object.keys(STAGES).length * (barH + 6) + 4;
      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'What the host and each GPU are doing over time, for the ' + S.name.toLowerCase() });
      const x = t => labelW + t / T * (W - labelW - 4), ly = i => top + i * (laneH + gap);
      el('text', { x: W, y: 13, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--ar-muted)' }, svg).textContent = 'time →';
      el('text', { x: labelW, y: 13, 'font-size': 12, fill: 'var(--ar-muted)' }, svg).textContent = '4 KB on eight A100s, stage fitted to width';
      for (let i = 0; i < lanes; i++) {
        el('text', { x: labelW - 8, y: ly(i) + laneH / 2 + 4, 'text-anchor': 'end', 'font-size': narrow ? 11 : 12.5, fill: 'var(--ar-ink2)' }, svg).textContent = i ? 'GPU ' + (i - 1) : 'host';
        el('rect', { x: labelW, y: ly(i), width: W - labelW - 4, height: laneH, fill: 'var(--ar-box)' }, svg);
      }
      ev.forEach(e => {
        const w = Math.max(1.5, x(e.t1) - x(e.t0) - 0.6);
        const r = el('rect', { x: x(e.t0), y: ly(e.lane) + 1, width: w, height: laneH - 2, rx: 1.5, fill: e.kind === 'issue' ? C.copy : C.ring }, svg);
        el('title', {}, r).textContent = e.title;
      });
      el('text', { x: narrow ? 84 : 112, y: totTop - 7, 'font-size': 12, fill: 'var(--ar-muted)' }, svg).textContent = 'Measured totals on one scale';
      const totW = narrow ? 84 : 112;   // wider than labelW so the stage names fit
      Object.keys(STAGES).forEach((k, i) => {
        const st = STAGES[k], y = totTop + i * (barH + 6), on = k === stage;
        const w = st.total / STAGES.host.total * (W - totW - 4 - (narrow ? 44 : 60));
        el('text', { x: totW - 8, y: y + barH - 1, 'text-anchor': 'end', 'font-size': narrow ? 10.5 : 11.5, fill: on ? 'var(--ar-ink)' : 'var(--ar-muted)', 'font-weight': on ? 'bold' : 'normal' }, svg).textContent = narrow ? st.name.split(' ')[0] : st.name;
        el('rect', { x: totW, y, width: Math.max(2, w), height: barH, rx: 1.5, fill: C.ring, opacity: on ? 1 : 0.3 }, svg);
        el('text', { x: totW + Math.max(2, w) + 6, y: y + barH - 1, 'font-size': 11.5, fill: on ? 'var(--ar-ink)' : 'var(--ar-muted)' }, svg).textContent = st.total + ' µs';
      });
      wrap.replaceChildren(svg);
      text.innerHTML = S.text;
    });
    tog.addEventListener('click', e => { const b = e.target.closest('button'); if (b) { stage = b.dataset.s; redraw(); } });
  }

  // ---------------------------------------------------------------- one card per machine: the path that matters, plus the numbers
  function mountMachine(fig) {
    const PCIE = '#7d858b', NV = '#567b65', PATH = '#376789', LIMIT = '#ad6245', SUM = '#567b65';
    const MACHINES = {
      t4: { W: 460, H: 270,
        facts: [
          ['GPUs', '2× T4, 40 SMs and 16 GB each'],
          ['Link', 'PCIe 3.0 x16 through a host bridge, spec about 16 GB/s each way'],
          ['Peer access', 'yes, in both directions'],
        ],
        draw(d) {
          d.box(70, 10, 320, 96, '', true);
          d.label(230, 30, 'host bridge', { anchor: 'middle', size: 13, color: '#333' });
          d.link('M130,106 V194 M330,106 V194', PCIE, 6);
          d.path('M122,194 V58 H338 V188', PATH);      // GPU 0 -> GPU 1
          d.path('M346,194 V78 H114 V188', '#817194'); // GPU 1 -> GPU 0
          d.label(230, 52, 'copy GPU 0 → GPU 1', { anchor: 'middle', size: 11.5, color: PATH });
          d.label(230, 96, 'copy GPU 1 → GPU 0', { anchor: 'middle', size: 11.5, color: '#817194' });
          d.label(142, 144, 'PCIe 3.0 x16', { size: 11.5, color: PCIE });
          d.label(142, 159, 'spec ≈16 GB/s each way', { size: 11.5, color: PCIE });
          d.box(60, 194, 140, 48, 'GPU 0'); d.box(260, 194, 140, 48, 'GPU 1');
          d.label(230, 264, 'measured: both copies together share 9.9 GB/s', { anchor: 'middle', size: 12.5, color: LIMIT });
        } },
      l4: { W: 460, H: 262,
        facts: [
          ['GPUs', '4× L4 (g6.12xlarge), 24 GB each'],
          ['Link', 'PCIe 4.0, spec up to 32 GB/s each way per x16 card'],
          ['Peer access', 'none, between any pair'],
          ['Every transfer', 'two staged copies through host memory'],
        ],
        draw(d) {
          d.box(40, 12, 120, 40, 'CPU'); d.box(270, 12, 150, 40, 'host memory', true);
          d.link('M160,32 H270', PCIE, 3);
          d.link('M100,52 V122 M70,122 H390 M70,122 V190 M170,122 V190 M290,122 V190 M390,122 V190', PCIE, 3);
          d.path('M58,190 V96 H312 V56', PATH);
          d.path('M378,56 V140 H182 V188', PATH);
          d.label(120, 90, '1. copy up', { size: 12, color: PATH });
          d.label(196, 160, '2. copy down', { size: 12, color: PATH });
          [30, 130, 250, 350].forEach((x, i) => d.box(x, 190, 80, 44, 'GPU ' + i));
          d.dashed('M332,212 H348', LIMIT);
          d.label(340, 256, '✕ no direct GPU-to-GPU copies', { anchor: 'middle', size: 12, color: LIMIT });
        } },
      a100: { W: 460, H: 286,
        facts: [
          ['GPUs', '8× A100 (p4d.24xlarge), 108 SMs and 40 GB each'],
          ['Link', 'NVLink through NVSwitch: 300 GB/s each way per GPU, shared across peers'],
          ['Peer access', 'every pair'],
        ],
        draw(d) {
          d.box(150, 8, 160, 34, 'host CPU');
          const gx = i => 14 + i * 55, gc = i => gx(i) + 23;
          d.link(`M230,42 V62 M${gc(0)},62 H${gc(7)}` + [0, 1, 2, 3, 4, 5, 6, 7].map(i => ` M${gc(i)},62 V84`).join(''), PCIE, 2);
          d.label(320, 30, 'PCIe: launches and copies', { size: 11.5, color: PCIE });
          for (let i = 0; i < 8; i++) d.link(`M${gc(i)},128 V200`, NV, 7);
          d.path(`M${gc(0) - 8},130 V186 H${gc(1) - 8} V134`, PATH);
          d.label(gc(1) + 6, 164, 'GPU 1 loads from GPU 0’s memory', { size: 11.5, color: PATH });
          for (let i = 0; i < 8; i++) d.box(gx(i), 84, 46, 44, 'GPU ' + i, false, 11.5);
          d.box(14, 200, 431, 38, 'NVSwitch');
          d.label(230, 262, 'NVLink: 300 GB/s each way per GPU', { anchor: 'middle', size: 12.5, color: NV });
          d.label(230, 280, 'vs PCIe 4.0 x16: 32 GB/s', { anchor: 'middle', size: 11.5, color: '#888' });
        } },
      hopper: { W: 460, H: 280,
        facts: [
          ['GPUs', '8× H100 or 8× B200'],
          ['Link', 'NVLink through NVSwitch: H100 450, B200 900 GB/s each way per GPU'],
          ['In-switch reduction (NVLS)', 'each GPU sends and receives a little over S, against 1.75S for an eight-GPU ring'],
        ],
        draw(d) {
          const gx = i => 14 + i * 55, gc = i => gx(i) + 23;
          for (let i = 0; i < 8; i++) { d.path(`M${gc(i) - 7},60 V164`, PATH); d.path(`M${gc(i) + 7},170 V66`, SUM); }
          for (let i = 0; i < 8; i++) d.box(gx(i), 16, 46, 44, 'GPU ' + i, false, 11.5);
          d.box(14, 170, 431, 50, '', true);
          d.label(230, 191, 'NVSwitch adds the inputs', { anchor: 'middle', size: 13, color: '#333' });
          d.label(230, 209, 'and multicasts the sum back', { anchor: 'middle', size: 12, color: '#555' });
          d.key(30, 246, PATH, 'each GPU’s input, to the switch');
          d.key(260, 246, SUM, 'the sum, back to every GPU');
          d.label(230, 272, 'NVLink: H100 450 / B200 900 GB/s each way per GPU', { anchor: 'middle', size: 12.5, color: NV });
        } },
    };
    const M = MACHINES[fig.dataset.machine];
    if (!M) return;
    const wrap = html('div', 'ar-machine');
    insertBeforeCaption(fig, wrap);
    const svg = el('svg', { viewBox: `0 0 ${M.W} ${M.H}`, role: 'img', style: `max-width:${M.W}px`,
      'aria-label': fig.querySelector('.ar-title') ? fig.querySelector('.ar-title').textContent : 'machine' }, wrap);
    const defs = el('defs', {}, svg), uid = fig.dataset.machine + '-' + Math.random().toString(36).slice(2, 6), marks = {};
    const marker = c => {
      if (marks[c]) return marks[c];
      const id = 'ar-mc-' + uid + '-' + Object.keys(marks).length;
      el('path', { d: 'M0,0 L10,5 L0,10 z', fill: c }, el('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto' }, defs));
      return (marks[c] = id);
    };
    const d = {
      box(x, y, w, h, t, hl, size) {
        el('rect', { x, y, width: w, height: h, fill: hl ? '#edf2f5' : '#f0f0f0', stroke: hl ? PATH : '#bbb', 'stroke-width': hl ? 1.5 : 1 }, svg);
        if (t) el('text', { x: x + w / 2, y: y + h / 2 + 4.5, 'text-anchor': 'middle', 'font-size': size || 13, fill: '#333' }, svg).textContent = t;
      },
      link(p, c, w) { el('path', { d: p, fill: 'none', stroke: c, 'stroke-width': w, opacity: 0.35 }, svg); },
      path(p, c) { el('path', { d: p, fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-linejoin': 'round', 'marker-end': `url(#${marker(c)})` }, svg); },
      dashed(p, c) { el('path', { d: p, fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-dasharray': '3 3' }, svg); },
      key(x, y, c, t) {
        el('path', { d: `M${x},${y - 4} H${x + 22}`, stroke: c, 'stroke-width': 2, 'marker-end': `url(#${marker(c)})` }, svg);
        d.label(x + 30, y, t, { size: 12 });
      },
      label(x, y, t, o) {
        o = o || {};
        el('text', { x, y, 'text-anchor': o.anchor || 'start', 'font-size': o.size || 12.5, fill: o.color || '#555',
          stroke: '#fafafa', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' }, svg).textContent = t;
      },
    };
    M.draw(d);
    const facts = html('dl', 'ar-facts', wrap);
    facts.innerHTML = M.facts.map(([k, v]) => '<dt>' + k + '</dt><dd>' + v + '</dd>').join('');
  }

  const MOUNT = { chart: mountChart, bars: mountBars, topology: mountTopology, ring: mountRing, model: mountModel, protocols: mountProtocols, allreduce: mountAllReduce, direct: mountDirect, movers: mountMovers, moverlimit: mountMoverLimit, machine: mountMachine, timeline: mountTimeline };
  function init() { document.querySelectorAll('figure.ar-fig[data-fig]').forEach(f => { const m = MOUNT[f.dataset.fig]; if (m) m(f); }); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
