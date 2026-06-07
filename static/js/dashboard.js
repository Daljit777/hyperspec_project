/* HyperVision — Global Dashboard JS */

// ── Stat counter animation ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.stat-value[data-target]').forEach(el => {
    const target = parseInt(el.dataset.target) || 0;
    if (target === 0) { el.textContent = '0'; return; }
    let current = 0;
    const step = Math.ceil(target / 40);
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current;
      if (current >= target) clearInterval(timer);
    }, 30);
  });

  // Sidebar collapse memory
  const collapsed = localStorage.getItem('sidebarCollapsed') === 'true';
  if (collapsed) document.getElementById('sidebar')?.classList.add('collapsed');
});

// Toggle sidebar with localStorage persistence
const origToggle = window.onclick;
document.getElementById('sidebarToggle')?.addEventListener('click', () => {
  const sb = document.getElementById('sidebar');
  sb.classList.toggle('collapsed');
  localStorage.setItem('sidebarCollapsed', sb.classList.contains('collapsed'));
});
