const cards = document.querySelectorAll('.metric[data-phase]');

document.querySelectorAll('.phase-switch button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.phase-switch button').forEach((item) => {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    cards.forEach((card) => {
      card.hidden = button.dataset.phase !== 'all' && card.dataset.phase !== button.dataset.phase;
    });
  });
});

document.querySelectorAll('.copy').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      const original = button.textContent;
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = original; }, 1200);
    } catch (_) {
      button.textContent = 'Select command';
    }
  });
});
