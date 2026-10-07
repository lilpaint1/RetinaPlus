// RetinaPlus — theme + language (i18n) engine, loaded on every page
(function () {
  const root = document.documentElement;

  // ---- theme ----
  const theme = localStorage.getItem('rp-theme') || 'light';
  root.setAttribute('data-theme', theme);
  window.toggleTheme = function () {
    const t = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', t);
    localStorage.setItem('rp-theme', t);
    paintTools();
  };

  // ---- language ----
  let lang = localStorage.getItem('rp-lang') || 'th';   // default Thai
  function applyLang() {
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-th]').forEach(el => {
      const v = el.getAttribute('data-' + lang);
      if (v != null) {
        if (el.hasAttribute('data-attr')) el.setAttribute(el.getAttribute('data-attr'), v);
        else el.textContent = v;
      }
    });
    document.title = (document.querySelector('[data-title-' + lang + ']')?.getAttribute('data-title-' + lang)) || document.title;
  }
  window.toggleLang = function () {
    lang = lang === 'th' ? 'en' : 'th';
    localStorage.setItem('rp-lang', lang);
    applyLang(); paintTools();
  };
  window.getLang = () => lang;

  // ---- tool buttons label ----
  function paintTools() {
    const t = root.getAttribute('data-theme');
    document.querySelectorAll('[data-tool="theme"] i').forEach(i =>
      i.className = 'ti ' + (t === 'dark' ? 'ti-sun' : 'ti-moon'));
    document.querySelectorAll('[data-tool="lang"]').forEach(b =>
      b.textContent = lang === 'th' ? 'EN' : 'ไทย');
  }

  document.addEventListener('DOMContentLoaded', function () { applyLang(); paintTools(); });
})();
