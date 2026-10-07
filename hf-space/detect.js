const $ = s => document.querySelector(s);
const STAGES = { th: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"], en: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"] };
const TIPS = {
  th: { 0: 'ไม่พบลักษณะของเบาหวานขึ้นจอตา', 1: 'พบสัญญาณเริ่มต้น เช่น microaneurysm เล็กน้อย', 2: 'พบลักษณะระดับปานกลาง เช่น hard exudate / จุดเลือดออก', 3: 'พบความผิดปกติระดับรุนแรง ควรพบจักษุแพทย์โดยเร็ว', 4: 'พบลักษณะระยะลุกลาม ควรพบจักษุแพทย์ทันที' },
  en: { 0: 'No signs of diabetic retinopathy.', 1: 'Early signs such as a few microaneurysms.', 2: 'Moderate features such as hard exudates / hemorrhages.', 3: 'Severe abnormalities. See an ophthalmologist soon.', 4: 'Proliferative features. See an ophthalmologist now.' }
};
const T = (th, en) => getLang() === 'en' ? en : th;

fetch('/status').then(r => r.json()).then(s => { if (s.demo) $('#modeTag').style.display = 'inline-flex'; }).catch(() => {});
setTimeout(() => { const s = document.getElementById('splash'); if (s) s.style.display = 'none'; }, 2500);

// ---- image source mode ----
let MODE = 'fundus';
document.querySelectorAll('.mode-opt').forEach(b => b.onclick = () => {
  document.querySelectorAll('.mode-opt').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); MODE = b.dataset.mode;
});

// ---- tabs ----
let stream = null, facing = 'environment';
$('#tabUp').onclick = () => setTab('up');
$('#tabCam').onclick = () => setTab('cam');
function setTab(t) {
  $('#tabUp').classList.toggle('active', t === 'up');
  $('#tabCam').classList.toggle('active', t === 'cam');
  $('#upPanel').style.display = t === 'up' ? 'block' : 'none';
  $('#camPanel').style.display = t === 'cam' ? 'block' : 'none';
  if (t === 'cam') startCam(); else stopCam();
}

// ---- upload ----
const drop = $('#drop'), file = $('#file');
$('#pickBtn').onclick = e => { e.stopPropagation(); file.click(); };
drop.onclick = e => { if (e.target.id !== 'pickBtn') file.click(); };
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });
file.onchange = () => { if (file.files[0]) analyze(file.files[0]); };

// ---- camera ----
async function startCam() {
  try {
    stopCam();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false });
    $('#video').srcObject = stream;
  } catch (e) {
    alert(T('เปิดกล้องไม่ได้ — ลองอนุญาตสิทธิ์กล้อง หรือใช้การอัปโหลดแทน', 'Cannot open camera — allow permission or use upload.'));
    setTab('up');
  }
}
function stopCam() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } }
$('#flip').onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; startCam(); };
$('#snap').onclick = () => {
  const v = $('#video'), c = $('#canvas');
  if (!v.videoWidth) return;
  const s = Math.min(v.videoWidth, v.videoHeight);
  c.width = s; c.height = s;
  c.getContext('2d').drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, s, s);
  c.toBlob(b => analyze(b), 'image/jpeg', 0.92);
};

// ---- analyze ----
function show(el) {
  ['#chooser', '#modeRow', '#upPanel', '#camPanel', '#loading', '#result'].forEach(s => $(s).style.display = 'none');
  if (el === '#result') $(el).style.display = 'flex';
  else if (el === '#loading') $(el).style.display = 'block';
  else { $('#modeRow').style.display = 'flex'; $('#chooser').style.display = 'grid'; $(el).style.display = 'block'; }
}
let lastFile = null, lastResult = null, lesionLoaded = false;
async function analyze(f) {
  stopCam(); show('#loading'); lastFile = f;
  const fd = new FormData(); fd.append('image', f); fd.append('mode', MODE);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    if (d.not_gradable) renderReject(d); else render(d);
  } catch (e) { alert(T('วิเคราะห์ไม่สำเร็จ: ', 'Failed: ') + e.message); backToChoose(); }
}

function setParts(on) {
  ['.result-main', '.disclaimer'].forEach(s => {
    const e = document.querySelector('#result ' + s); if (e) e.style.display = on ? '' : 'none';
  });
}

function renderReject(d) {
  const v = $('#verdict');
  v.className = 'card verdict warn';
  v.innerHTML = `<div class="vi"><i class="ti ti-photo-x"></i></div>
    <div class="vt"><h2>${T('ภาพใช้วินิจฉัยไม่ได้', 'Image not gradable')}</h2>
      <p>${getLang() === 'en' ? d.message_en : d.message_th}</p></div>`;
  setParts(false);
  show('#result');
}

// ---- viewer state ----
let LAYERS = {}, baseSel = 'raw';
function applyOpacity() { $('#imgLesion').style.opacity = ($('#opacity').value / 100).toFixed(2); }
function setBaseMode(m) {
  if (m === 'enh' && !LAYERS.enhanced) return;
  baseSel = m;
  $('#baseRaw').classList.toggle('on', m === 'raw');
  $('#baseEnh').classList.toggle('on', m === 'enh');
  $('#imgBase').src = 'data:image/png;base64,' + (m === 'enh' ? LAYERS.enhanced : LAYERS.raw);
  $('#baseLbl').textContent = m === 'enh' ? T('ภาพปรับชัด (CLAHE)', 'Enhanced (CLAHE)') : T('ภาพต้นฉบับ (Raw)', 'Original (Raw)');
}
function toggleLayer(btn, img) {
  if (btn.classList.contains('disabled')) return;
  const on = btn.classList.toggle('on');
  img.style.visibility = on ? 'visible' : 'hidden';
}
$('#tgLesion').onclick = () => toggleLayer($('#tgLesion'), $('#imgLesion'));
$('#tgLandmark').onclick = () => toggleLayer($('#tgLandmark'), $('#imgLandmark'));
$('#baseRaw').onclick = () => setBaseMode('raw');
$('#baseEnh').onclick = () => setBaseMode('enh');
$('#opacity').oninput = applyOpacity;

function setupViewer(d) {
  LAYERS = d.layers || {};                   // มีแค่ raw ตอนแรก — lesion โหลดทีหลังผ่าน /explain
  const les = $('#imgLesion'), lm = $('#imgLandmark');
  setBaseMode('raw');
  les.style.display = 'none'; les.style.visibility = 'hidden';
  lm.style.display = 'none'; lm.style.visibility = 'hidden';
  $('#tgLesion').classList.add('disabled'); $('#tgLesion').classList.remove('on');
  $('#tgLandmark').classList.add('disabled'); $('#tgLandmark').classList.remove('on');
  $('#baseEnh').classList.add('disabled');
  $('#opacity').parentElement.style.display = 'none';
  applyOpacity();
}

async function loadExplain() {
  lesionLoaded = true;                       // กันยิงซ้ำ
  $('#findings').innerHTML = `<div class="lesion-loading"><i class="ti ti-loader-2 spin"></i> ${T('กำลังวิเคราะห์รอยโรค…', 'Analyzing lesions…')}</div>`;
  try {
    const fd = new FormData(); fd.append('image', lastFile); fd.append('mode', MODE);
    fd.append('grade_name', (lastResult && lastResult.grade_name) || '');
    const ex = await fetch('/explain', { method: 'POST', body: fd }).then(r => r.json());
    if (ex.error || ex.lesion_unavailable) {
      $('#findings').innerHTML = `<div class="frow"><span class="fl">${T('รอยโรค', 'Lesions')}</span><span class="fv" style="font-weight:400">${T('วิเคราะห์ไม่ได้', 'unavailable')}</span></div>`;
      return;
    }
    LAYERS.lesion = ex.layers.lesion; LAYERS.landmark = ex.layers.landmark; LAYERS.enhanced = ex.layers.enhanced;
    const les = $('#imgLesion'), lm = $('#imgLandmark');
    les.src = 'data:image/png;base64,' + ex.layers.lesion; les.style.display = '';
    lm.src = 'data:image/png;base64,' + ex.layers.landmark; lm.style.display = '';
    $('#tgLesion').classList.remove('disabled'); $('#tgLesion').classList.add('on');
    $('#tgLandmark').classList.remove('disabled'); $('#tgLandmark').classList.add('on');
    $('#baseEnh').classList.remove('disabled');
    $('#opacity').parentElement.style.display = '';
    $('#imgLesion').style.visibility = 'visible'; $('#imgLandmark').style.visibility = 'visible';
    applyOpacity();
    renderFindings(ex);
    $('#why').innerHTML = `<i class="ti ti-bulb"></i><div class="wt"><b>${T('เหตุผลจาก AI', 'AI rationale')}:</b> ${ex.rationale}</div>`;
  } catch (e) {
    $('#findings').innerHTML = `<div class="frow"><span class="fl">error</span><span class="fv" style="font-weight:400">${e.message}</span></div>`;
  }
}

let deepOpened = false;
$('#deepBtn').onclick = async () => {
  const dd = $('#deepdive'), open = dd.style.display === 'none';
  dd.style.display = open ? 'block' : 'none';
  $('#deepBtn').classList.toggle('open', open);
  if (open) {
    if (lastResult && lastResult.lesion_available && !lesionLoaded && lastFile) await loadExplain();
    $('#imgLesion').style.visibility = $('#tgLesion').classList.contains('on') ? 'visible' : 'hidden';
    $('#imgLandmark').style.visibility = $('#tgLandmark').classList.contains('on') ? 'visible' : 'hidden';
    applyOpacity();
    dd.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {                                   // collapse → clean screener image
    $('#imgLesion').style.visibility = 'hidden';
    $('#imgLandmark').style.visibility = 'hidden';
  }
};

function renderFindings(d) {
  const f = d.features, fov = d.fovea_metrics;
  if (!f) {
    const msg = (lastResult && lastResult.lesion_available)
      ? T('เปิดส่วนนี้เพื่อให้ AI วิเคราะห์รอยโรค', 'AI analyzes lesions on open')
      : T('การวิเคราะห์รอยโรคไม่พร้อมใช้งาน', 'lesion analysis unavailable');
    $('#findings').innerHTML = `<div class="frow"><span class="fl">${T('รอยโรค', 'Lesions')}</span><span class="fv" style="font-weight:400;color:var(--text-3)">${msg}</span></div>`;
    return;
  }
  const exdd = (fov && fov.EX && fov.EX.min_dd != null) ? fov.EX.min_dd : null;
  const exWarn = exdd != null && exdd <= 1.0;
  const chip = (txt, k) => txt ? `<span class="fchip ${k}">${txt}</span>` : '';
  // ระดับจากจำนวน: <=a เล็กน้อย, <=b ปานกลาง, >b มาก
  const sev = (n, a, b) => n === 0 ? ['', ''] : (n <= a ? [T('เล็กน้อย', 'mild'), 'ok'] : n <= b ? [T('ปานกลาง', 'moderate'), 'warn'] : [T('มาก', 'many'), 'bad']);
  const rows = [];
  let s;
  s = sev(f.MA.count, 15, 40);
  rows.push([T('จุดโป่งหลอดเลือด', 'Microaneurysm'), 'Microaneurysm', f.MA.count ? `${f.MA.count} ${T('จุด', 'spots')}` : T('ไม่พบ', 'none'), s[0], s[1]]);
  s = sev(f.HE.count, 5, 30);
  rows.push([T('จุดเลือดออก', 'Hemorrhage'), 'Hemorrhage', f.HE.count ? `${f.HE.count} ${T('จุด', 'spots')}` : T('ไม่พบ', 'none'), s[0], s[1]]);
  if (f.EX.area_px > 0) {
    const ddtxt = exdd != null ? `${T('ห่างจุดรับภาพ', 'from fovea')} ${exdd.toFixed(1)} DD` : T('พบ', 'present');
    rows.push([T('คราบไขมัน', 'Hard exudate'), 'Hard exudate', ddtxt,
      exWarn ? T('⚠ เสี่ยงตรงจุดรับภาพ', '⚠ near fovea') : T('ไม่ใกล้จุดรับภาพ', 'not near fovea'), exWarn ? 'bad' : 'ok']);
  } else rows.push([T('คราบไขมัน', 'Hard exudate'), 'Hard exudate', T('ไม่พบ', 'none'), '', '']);
  rows.push([T('ปุยสำลี', 'Soft exudate'), 'Soft exudate', f.SE.area_px > 0 ? T('พบ', 'present') : T('ไม่พบ', 'none'), '', f.SE.area_px > 0 ? 'warn' : '']);

  let html = rows.map(r => `<div class="frow"><span class="fl">${r[0]} <small>${r[1]}</small></span><span class="fv">${r[2]} ${chip(r[3], r[4])}</span></div>`).join('');
  if (exWarn) html += `<div class="warnbox"><i class="ti ti-alert-circle"></i> ${T('คราบไขมันอยู่ใกล้จุดรับภาพชัด (< 1 เท่าจานประสาทตา) — เสี่ยงจอบวมน้ำ', 'Exudate within 1 disc-diameter of the fovea — macular edema risk')}</div>`;
  $('#findings').innerHTML = html;
}

const CONF = { normal: { th: 'สูง', en: 'high' }, review: { th: 'ปานกลาง', en: 'medium' }, refer: { th: 'สูง', en: 'high' } };
function render(d) {
  setParts(true);
  lastResult = d; lesionLoaded = false;
  $('#deepdive').style.display = 'none'; $('#deepBtn').classList.remove('open'); deepOpened = false;
  // binary action สำหรับคนคัดกรอง: review รวมเข้า "ควรพบแพทย์" (ไม่มั่นใจ = ให้หมอดู, ทิศปลอดภัย)
  const needDoc = d.zone !== 'normal';
  const zc = needDoc ? (d.zone === 'refer' ? 'bad' : 'warn') : 'ok';
  const zi = needDoc ? (d.zone === 'refer' ? 'alert-triangle' : 'alert-circle') : 'circle-check';
  const title = needDoc
    ? (d.grade >= 3 ? T('ควรพบจักษุแพทย์โดยเร็ว', 'See an ophthalmologist soon') : T('ควรพบจักษุแพทย์', 'See an ophthalmologist'))
    : T('ไม่ต้องส่งต่อ', 'No referral needed');
  const conf = getLang() === 'en' ? CONF[d.zone].en : CONF[d.zone].th;
  const v = $('#verdict');
  v.className = 'card verdict ' + zc;
  v.innerHTML = `<div class="vi"><i class="ti ti-${zi}"></i></div>
    <div class="vt"><h2>${title}</h2>
      <p>${T('ผลคัดกรอง', 'Result')}: <b>${d.grade_name}</b> · ${T('ความมั่นใจ', 'confidence')} ${conf}${d.demo ? ' · ' + T('สาธิต', 'demo') : ''}</p></div>
    <div class="vp"><b>${d.risk_percent}%</b><span>${T('ความเสี่ยง', 'risk')}</span></div>`;

  const names = STAGES[getLang()];
  $('#stages').innerHTML = names.map((s, i) => `<div class="st ${i === d.grade ? 'on g' + i : ''}">${i === 4 ? 'Prolif.' : s}</div>`).join('');
  $('#scoreFill').style.width = Math.min(100, Math.max(3, d.score / 4 * 100)) + '%';
  $('#scoreVal').textContent = 'score ' + d.score.toFixed(2);

  setupViewer(d);
  renderFindings(d);
  $('#why').innerHTML = d.rationale
    ? `<i class="ti ti-bulb"></i><div class="wt"><b>${T('เหตุผลจาก AI', 'AI rationale')}:</b> ${d.rationale}</div>`
    : `<i class="ti ti-bulb"></i><div class="wt">${TIPS[getLang()][d.grade]}</div>`;
  show('#result');
}

function backToChoose() {
  ['#loading', '#result', '#camPanel'].forEach(s => $(s).style.display = 'none');
  $('#modeRow').style.display = 'flex'; $('#chooser').style.display = 'grid'; $('#upPanel').style.display = 'block';
  $('#tabUp').classList.add('active'); $('#tabCam').classList.remove('active');
}
$('#again').onclick = () => { file.value = ''; backToChoose(); };
$('#send').onclick = () => alert(T('ส่งผลให้จักษุแพทย์ (เชื่อมระบบโรงพยาบาลในเฟสถัดไป)', 'Sent to ophthalmologist (hospital integration next phase)'));
