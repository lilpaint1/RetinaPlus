const STAGES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"];

Promise.all([
  fetch('/stats').then(r => r.json()),
  fetch('/history').then(r => r.json())
]).then(([s, h]) => {
  document.getElementById('mTotal').textContent = s.total;
  document.getElementById('mRefer').textContent = s.referable;
  document.getElementById('mNormal').textContent = s.normal;
  document.getElementById('mRate').textContent = (s.total ? Math.round(s.referable / s.total * 100) : 0) + '%';

  const max = Math.max(1, ...STAGES.map(c => s.dist[c] || 0));
  document.getElementById('dist').innerHTML = STAGES.map((c, i) => {
    const v = s.dist[c] || 0;
    return `<div class="row g${i}"><div class="top"><span>${c}</span><b>${v}</b></div>
      <div class="bar"><i style="width:${v / max * 100}%"></i></div></div>`;
  }).join('');

  const hist = document.getElementById('hist');
  if (!h.length) { hist.innerHTML = '<div class="empty">ยังไม่มีประวัติการตรวจ</div>'; return; }
  hist.innerHTML = h.slice(0, 30).map(x => `
    <div class="h-row"><span class="h-dot ${x.referable ? 'bad' : 'ok'}"></span>
      <span class="g">${x.grade_name}</span>
      <span class="t">${(x.time || '').replace('T', ' ')}</span>
      <span class="r" style="color:${x.referable ? 'var(--danger)' : 'var(--brand)'}">${x.risk_percent}%</span></div>`).join('');
}).catch(() => {
  document.getElementById('hist').innerHTML = '<div class="empty">โหลดข้อมูลไม่สำเร็จ</div>';
});
