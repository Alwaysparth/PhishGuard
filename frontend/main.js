/* ============================================================
   PhishGuard — main.js
   Shared JavaScript for all pages
   ============================================================
 
   DEPLOYMENT INSTRUCTIONS:
   ─────────────────────────────────────────────────────────
   Change API_BASE to your Render backend URL once deployed.
 
   Local dev:
     const API_BASE = 'http://localhost:8000';
 
   After deploying to Render:
     const API_BASE = 'https://phishguard-api.onrender.com';
   ─────────────────────────────────────────────────────────
*/
 
'use strict';
 
/* ── API Base URL — UPDATE THIS after Render deployment ────── */
const API_BASE = 'https://phishguard-api-n4tj.onrender.com'
  || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'                      // local dev
  : 'https://phishguard-api-n4tj.onrender.com';         // ← replace with your real Render URL
 
/* ── Navbar HTML ─────────────────────────────────────────────── */
const NAV_HTML = `
<nav id="navbar">
  <a href="index.html" class="nav-brand">
    <svg class="nav-brand-icon" viewBox="0 0 32 32" fill="none">
      <path d="M16 2L4 8v8c0 6.6 5.1 12.8 12 14.3C22.9 28.8 28 22.6 28 16V8L16 2z"
            stroke="currentColor" stroke-width="2" fill="rgba(0,255,170,0.10)"/>
      <path d="M10 16l4 4 8-8" stroke="currentColor" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    PHISHGUARD
  </a>
  <div class="nav-links" id="nav-links">
    <a href="index.html"     class="nav-link" data-page="home">Home</a>
    <a href="scanner.html"   class="nav-link" data-page="scanner">Scanner</a>
    <a href="workflow.html"  class="nav-link" data-page="workflow">Workflow</a>
    <a href="analytics.html" class="nav-link" data-page="analytics">Analytics</a>
    <a href="team.html"      class="nav-link" data-page="team">Team</a>
    <a href="contact.html"   class="nav-link" data-page="contact">Contact</a>
  </div>
  <button class="nav-burger" id="nav-burger" aria-label="Toggle menu">
    <span></span><span></span><span></span>
  </button>
</nav>
<div class="nav-mobile" id="nav-mobile">
  <a href="index.html"     class="nav-link" data-page="home">Home</a>
  <a href="scanner.html"   class="nav-link" data-page="scanner">Scanner</a>
  <a href="workflow.html"  class="nav-link" data-page="workflow">Workflow</a>
  <a href="analytics.html" class="nav-link" data-page="analytics">Analytics</a>
  <a href="team.html"      class="nav-link" data-page="team">Team</a>
  <a href="contact.html"   class="nav-link" data-page="contact">Contact</a>
</div>`;
 
const FOOTER_HTML = `
<footer>
  <a href="index.html" class="footer-brand">PHISHGUARD</a>
  <span class="footer-copy">© 2026 PhishGuard — AI-Powered Phishing Detection</span>
  <div class="footer-links">
    <a href="scanner.html"  class="footer-link">Scanner</a>
    <a href="workflow.html" class="footer-link">Workflow</a>
    <a href="contact.html"  class="footer-link">Contact</a>
  </div>
</footer>`;
 
/* ── Inject nav + footer ─────────────────────────────────────── */
function initLayout() {
  const navHolder = document.getElementById('nav-holder');
  if (navHolder) navHolder.innerHTML = NAV_HTML;
 
  const footerHolder = document.getElementById('footer-holder');
  if (footerHolder) footerHolder.innerHTML = FOOTER_HTML;
 
  // Highlight active nav link
  const page = document.body.dataset.page;
  document.querySelectorAll(`.nav-link[data-page="${page}"]`)
    .forEach(el => el.classList.add('active'));
 
  // Mobile burger
  const burger     = document.getElementById('nav-burger');
  const mobileMenu = document.getElementById('nav-mobile');
  if (burger && mobileMenu) {
    burger.addEventListener('click', () => {
      const open = mobileMenu.classList.toggle('open');
      burger.classList.toggle('open', open);
    });
    mobileMenu.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        mobileMenu.classList.remove('open');
        burger.classList.remove('open');
      });
    });
  }
}
 
/* ── Scroll reveal ───────────────────────────────────────────── */
function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const delay = entry.target.dataset.delay || 0;
        setTimeout(() => entry.target.classList.add('visible'), delay);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
 
  document.querySelectorAll('.reveal').forEach((el, i) => {
    if (!el.dataset.delay) el.dataset.delay = i * 60;
    observer.observe(el);
  });
}
 
/* ── Toast ───────────────────────────────────────────────────── */
function showToast(msg, type = 'success') {
  let toast = document.getElementById('pg-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'pg-toast';
    toast.style.cssText = `
      position:fixed; bottom:32px; right:32px; z-index:999;
      background:var(--bg-elevated); border:1px solid var(--border-mid);
      font-family:var(--mono); font-size:13px; letter-spacing:1px;
      padding:14px 20px; border-radius:8px;
    `;
    document.body.appendChild(toast);
  }
  toast.style.color = type === 'error' ? 'var(--red)'
                    : type === 'warn'  ? 'var(--yellow)'
                    : 'var(--green)';
  toast.textContent = msg;
  toast.style.display = 'block';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { toast.style.display = 'none'; }, 3500);
}
 
/* ── API helpers ─────────────────────────────────────────────── */
async function checkURL(url, mode = 'viewer') {
  const res = await fetch(`${API_BASE}/check-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, mode }),
    signal: AbortSignal.timeout(12000),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
 
async function fetchAnalytics() {
  const res = await fetch(`${API_BASE}/api/v1/analytics/summary`, {
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
 
/* ── Demo fallback ───────────────────────────────────────────── */
function demoResult(url) {
  const u = url.toLowerCase();
  if (u.includes('paypal-secure') || (u.includes('verify') && u.includes('.tk'))
      || u.includes('amazon-prize') || u.includes('phishing')) {
    return { risk_score: 94, status: 'phishing', action: 'none',
      reasons: ['Brand impersonation detected', 'Suspicious TLD (.tk/.xyz)',
                'Login keywords in URL', 'Encoded characters found'] };
  }
  if (u.includes('-secure') || u.includes('login') && !u.startsWith('https://www')) {
    return { risk_score: 68, status: 'suspicious', action: 'none',
      reasons: ['Security keyword in URL path', 'Moderate risk indicators detected'] };
  }
  return { risk_score: 5, status: 'safe', action: 'none',
    reasons: ['No suspicious patterns detected', 'Domain structure appears legitimate'] };
}
 
/* ── Init ────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initLayout();
  initReveal();
});