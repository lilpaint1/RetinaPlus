const $ = s => document.querySelector(s);
const STAGES = { th: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"], en: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"] };
const TIPS = {
  th: {
    0: 'ไม่พบลักษณะของเบาหวานขึ้นจอตา จอตาและเส้นเลือดอยู่ในเกณฑ์ปกติ',
    1: 'พบสัญญาณเริ่มต้น เช่น microaneurysm เล็กน้อย แนะนำติดตามอาการ',
    2: 'พบลักษณะระดับปานกลาง เช่น hard exudate / จุดเลือดออก',
    3: 'พบความผิดปกติระดับรุนแรง ควรพบจักษุแพทย์โดยเร็ว',
    4: 'พบลักษณะระยะลุกลาม (เส้นเลือดงอกผิดปกติ) ควรพบจักษุแพทย์ทันที'
  },
  en: {
    0: 'No signs of diabetic retinopathy. Retina and vessels appear normal.',
    1: 'Early signs such as a few microaneurysms. Monitoring recommended.',
    2: 'Moderate features such as hard exudates / hemorrhages.',
    3: 'Severe abnormalities. See an ophthalmologist soon.',
    4: 'Proliferative features (abnormal new vessels). See an ophthalmologist now.'
  }
};
const T = (th, en) => getLang() === 'en' ? en : th;

fetch('/status').then(r => r.json()).then(s => { if (s.demo) $('#modeTag').style.display = 'inline-flex'; }).catch(() => {});

// dismiss splash (so it ไม่บัง clicks)
setTimeout(() => { const s = document.getElementById('splash'); if (s) s.style.display = 'none'; }, 2500);

// ---- tabs ----
let stream = null, facing = 'environment';
$('#tabUp').onclick = () => { setTab('up'); };
$('#tabCam').onclick = () => { setTab('cam'); };
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
    alert(T('เปิดกล้องไม่ได้ — ลองอนุญาตสิทธิ์กล้อง หรือใช้การอัปโหลดแทน', 'Cannot open camera — allow camera permission or use upload.'));
    setTab('up');
  }
}
function stopCam() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } }
$('#flip').onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; startCam(); };
$('#snap').onclick = () => {
  const v = $('#video'), c = $('#canvas');
  if (!v.videoWidth) return;
  const s = Math.min(v.videoWidth, v.videoHeight);            // crop กลางเป็นสี่เหลี่ยม
  c.width = s; c.height = s;
  c.getContext('2d').drawImage(v, (v.videoWidth - s) / 2, (v.videoHeight - s) / 2, s, s, 0, 0, s, s);
  c.toBlob(b => analyze(b), 'image/jpeg', 0.92);
};

// ---- analyze ----
function show(el) {
  ['#chooser', '#upPanel', '#camPanel', '#loading', '#result'].forEach(s => $(s).style.display = 'none');
  if (el === '#result') $(el).style.display = 'flex';
  else if (el === '#loading') $(el).style.display = 'block';
  else { $('#chooser').style.display = 'grid'; $(el).style.display = el === '#camPanel' ? 'block' : 'block'; }
}
async function analyze(f) {
  stopCam(); show('#loading');
  const fd = new FormData(); fd.append('image', f);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    if (d.not_fundus) renderNotFundus(d); else render(d);
  } catch (e) { alert(T('วิเคราะห์ไม่สำเร็จ: ', 'Failed: ') + e.message); backToChoose(); }
}

function setParts(on) {
  ['.stage-card', '.img-grid', '.why', '.disclaimer'].forEach(s => {
    const e = document.querySelector('#result ' + s); if (e) e.style.display = on ? '' : 'none';
  });
}

function renderNotFundus(d) {
  const v = $('#verdict');
  v.className = 'card verdict warn';
  v.innerHTML = `<div class="vi"><i class="ti ti-photo-x"></i></div>
    <div class="vt"><h2>${T('ไม่ใช่ภาพจอประสาทตา', 'Not a fundus image')}</h2>
      <p>${getLang() === 'en' ? d.message_en : d.message_th}</p></div>`;
  setParts(false);
  show('#result');
}

function render(d) {
  setParts(true);
  const v = $('#verdict');
  v.className = 'card verdict ' + (d.referable ? 'bad' : 'ok');
  v.innerHTML = `<div class="vi"><i class="ti ti-${d.referable ? 'alert-triangle' : 'circle-check'}"></i></div>
    <div class="vt"><h2>${d.referable ? T('พบความผิดปกติ — ควรพบจักษุแพทย์', 'Abnormal — see an ophthalmologist') : T('ปกติ — ไม่พบความเสี่ยงที่ต้องส่งต่อ', 'Normal — no referral needed')}</h2>
      <p>${T('ระดับ', 'Level')}: <b>${d.grade_name}</b>${d.demo ? ' · ' + T('สาธิต', 'demo') : ''}</p></div>
    <div class="vp"><b>${d.risk_percent}%</b><span>${T('ความเสี่ยง', 'risk')}</span></div>`;

  const names = STAGES[getLang()];
  $('#stages').innerHTML = names.map((s, i) => `<div class="st ${i === d.grade ? 'on g' + i : ''}">${i === 4 ? 'Prolif.' : s}</div>`).join('');
  $('#scoreFill').style.width = Math.min(100, Math.max(3, d.score / 4 * 100)) + '%';
  $('#scoreVal').textContent = 'score ' + d.score.toFixed(2);
  if (d.preprocessed) $('#imgPre').src = 'data:image/png;base64,' + d.preprocessed;
  if (d.gradcam) { $('#imgCam').src = 'data:image/png;base64,' + d.gradcam; $('#imgCam').parentElement.style.display = 'block'; }
  else $('#imgCam').parentElement.style.display = 'none';
  $('#why').innerHTML = `<i class="ti ti-bulb"></i><div class="wt"><b>${T('AI พิจารณาอะไร', 'What the AI considered')}:</b> ${TIPS[getLang()][d.grade]} ${T('ภาพด้านบนแสดงบริเวณที่โมเดลให้ความสำคัญ', 'The image above shows where the model focused.')}</div>`;
  show('#result');
}

function backToChoose() {
  ['#loading', '#result', '#camPanel'].forEach(s => $(s).style.display = 'none');
  $('#chooser').style.display = 'grid'; $('#upPanel').style.display = 'block';
  $('#tabUp').classList.add('active'); $('#tabCam').classList.remove('active');
}
$('#again').onclick = () => { file.value = ''; backToChoose(); };
$('#send').onclick = () => alert(T('ส่งผลให้จักษุแพทย์ (เชื่อมระบบโรงพยาบาลในเฟสถัดไป)', 'Sent to ophthalmologist (hospital integration in next phase)'));