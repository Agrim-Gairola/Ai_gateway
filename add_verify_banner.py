f = open('gateway/dashboard.html', encoding='utf-8')
content = f.read()
f.close()

old = 'async function loadUserNav() {'
new = '''async function loadUserNav() {
  try {
    const token = localStorage.getItem('token');
    const h = token
      ? {'X-API-Key': API_KEY, 'Authorization': 'Bearer ' + token}
      : {'X-API-Key': API_KEY};
    const r = await fetch('/auth/me', {headers: h});
    if (r.ok) {
      const u = await r.json();
      if (!u.verified) {
        const banner = document.createElement('div');
        banner.style.cssText = 'background:rgba(245,158,11,.1);border-bottom:1px solid rgba(245,158,11,.3);padding:10px 40px;font-size:13px;color:var(--amber);display:flex;align-items:center;justify-content:space-between;position:relative;z-index:200';
        banner.innerHTML = `
          <span>Please verify your email address to unlock all features.</span>
          <button onclick="resendVerification()" style="background:rgba(245,158,11,.2);border:1px solid rgba(245,158,11,.4);color:var(--amber);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-family:JetBrains Mono,monospace">Resend email</button>
        `;
        document.querySelector('header').insertAdjacentElement('afterend', banner);
      }
    }
  } catch(e) {}
}

async function resendVerification() {
  try {
    const token = localStorage.getItem('token');
    const h = token
      ? {'X-API-Key': API_KEY, 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
      : {'X-API-Key': API_KEY, 'Content-Type': 'application/json'};
    await fetch('/auth/resend-verification', {method: 'POST', headers: h});
    alert('Verification email sent. Please check your inbox.');
  } catch(e) {}
}

async function loadUserNav_old() {'''

if 'async function loadUserNav() {' in content:
    content = content.replace('async function loadUserNav() {', new, 1)
    # Clean up the old function since we replaced it
    f = open('gateway/dashboard.html', 'w', encoding='utf-8')
    f.write(content)
    f.close()
    print('Done')
else:
    print('Not found')