// reveal on scroll
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.style.animationPlayState = 'running'; io.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach(el => {
  if (el.getBoundingClientRect().top > window.innerHeight) {
    el.style.animationPlayState = 'paused';
    io.observe(el);
  }
});
