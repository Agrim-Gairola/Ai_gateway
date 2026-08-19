f = open('gateway/login.html', encoding='utf-8')
content = f.read()
f.close()

old = '</script>\n</body>'

new = '''
// Forgot password flow
const urlParams = new URLSearchParams(window.location.search);
const resetToken = urlParams.get('reset_token');
if (window.location.pathname === '/reset-password' || resetToken) {
  document.querySelector('.tabs').style.display = 'none';
  document.getElementById('form-login').style.display = 'none';
  document.getElementById('form-signup').style.display = 'none';
  document.querySelector('.google-btn').style.display = 'none';
  document.querySelector('.divider').style.display = 'none';
  document.querySelector('h1') && (document.querySelector('h1').textContent = 'Reset password');
  const resetForm = document.createElement('div');
  resetForm.innerHTML = `
    <div class="form-group">
      <label>New Password</label>
      <input type="password" id="new-password" placeholder="Min 8 characters" />
    </div>
    <div class="form-group">
      <label>Confirm Password</label>
      <input type="password" id="confirm-password" placeholder="Repeat password" />
    </div>
    <button class="btn btn-primary" onclick="doReset()">Reset Password</button>
    <div class="error" id="reset-error"></div>
  `;
  document.querySelector('.card').appendChild(resetForm);
}

async function doReset() {
  const token = new URLSearchParams(window.location.search).get('token');
  const newPw = document.getElementById('new-password').value;
  const confirmPw = document.getElementById('confirm-password').value;
  const err = document.getElementById('reset-error');
  if (newPw !== confirmPw) { showError(err, 'Passwords do not match'); return; }
  if (newPw.length < 8) { showError(err, 'Password must be at least 8 characters'); return; }
  try {
    const r = await fetch('/auth/reset-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({token, new_password: newPw})
    });
    const d = await r.json();
    if (!r.ok) { showError(err, d.detail || 'Reset failed'); return; }
    window.location.href = '/login?reset=1';
  } catch(e) { showError(err, 'Connection error'); }
}

// Show forgot password link
const loginForm = document.getElementById('form-login');
if (loginForm) {
  const forgotLink = document.createElement('div');
  forgotLink.style.cssText = 'text-align:right;margin-top:8px';
  forgotLink.innerHTML = '<a href="#" onclick="showForgotPassword()" style="font-size:13px;color:var(--muted);text-decoration:none">Forgot password?</a>';
  loginForm.insertBefore(forgotLink, loginForm.querySelector('button'));
}

function showForgotPassword() {
  document.getElementById('form-login').style.display = 'none';
  const fp = document.createElement('div');
  fp.id = 'form-forgot';
  fp.innerHTML = `
    <p style="font-size:14px;color:var(--muted);margin-bottom:16px">Enter your email and we will send you a reset link.</p>
    <div class="form-group">
      <label>Email</label>
      <input type="email" id="forgot-email" placeholder="you@example.com" />
    </div>
    <button class="btn btn-primary" onclick="doForgot()">Send Reset Link</button>
    <div style="margin-top:12px"><a href="#" onclick="switchTab('login')" style="font-size:13px;color:var(--muted);text-decoration:none">Back to sign in</a></div>
    <div class="error" id="forgot-error"></div>
    <div id="forgot-success" style="color:var(--green);font-size:13px;margin-top:12px;display:none"></div>
  `;
  document.querySelector('.card').appendChild(fp);
}

async function doForgot() {
  const email = document.getElementById('forgot-email').value.trim();
  const err = document.getElementById('forgot-error');
  const success = document.getElementById('forgot-success');
  if (!email) { showError(err, 'Please enter your email'); return; }
  try {
    const r = await fetch('/auth/forgot-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email})
    });
    const d = await r.json();
    err.style.display = 'none';
    success.textContent = d.message;
    success.style.display = 'block';
  } catch(e) { showError(err, 'Connection error'); }
}

// Show verification banner if needed
const verified = new URLSearchParams(window.location.search).get('verified');
if (verified) {
  const banner = document.createElement('div');
  banner.style.cssText = 'background:rgba(34,211,165,.1);border:1px solid rgba(34,211,165,.3);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:var(--green);text-align:center';
  banner.textContent = 'Email verified successfully. You can now sign in.';
  document.querySelector('.card').insertBefore(banner, document.querySelector('.tabs'));
}
</script>
</body>'''

if '</script>\n</body>' in content:
    f = open('gateway/login.html', 'w', encoding='utf-8')
    f.write(content.replace('</script>\n</body>', new, 1))
    f.close()
    print('Done')
else:
    print('Not found - checking end of file')
    print(repr(content[-100:]))