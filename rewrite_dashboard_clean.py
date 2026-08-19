f = open('gateway/dashboard.html', encoding='utf-8')
content = f.read()
f.close()

# Find script start and end
script_start = content.find('<script>')
script_end = content.find('</script>') + len('</script>')

html_before = content[:script_start]
html_after = content[script_end:]

clean_script = """<script>
const API_KEY = 'ak_dev_1234567890abcdef';
let stats = {requests:0,totalCost:0,totalBaseline:0,totalLatency:0,success:0,models:{},totalCompressionSaved:0,totalOriginalTokens:0,totalCompressedTokens:0};
let spdStats = {consensus:0, escalated:0};
let costHistory = [];
let sparklineChart = null;
const MODEL_COLORS = {'llama-3.1-8b-instant':'#7c6af7','llama-3.3-70b-versatile':'#38bdf8','gemini-2.5-flash':'#22d3a5','gemini-2.0-flash':'#22d3a5'};
const GPT4O_COST_PER_TOKEN = 0.0000025;

function initSparkline() {
  const canvas = document.getElementById('cost-sparkline');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  sparklineChart = {ctx, canvas, draw() {
    const w = canvas.offsetWidth || 300;
    const h = 120;
    canvas.width = w; canvas.height = h;
    ctx.clearRect(0, 0, w, h);
    if (costHistory.length < 2) {
      ctx.fillStyle = '#6b6b8a'; ctx.font = '11px JetBrains Mono';
      ctx.textAlign = 'center';
      ctx.fillText('Send requests to see cost trend', w/2, h/2);
      return;
    }
    const max = Math.max(...costHistory) * 1.2 || 0.001;
    const pad = 20;
    const points = costHistory.map((v, i) => ({
      x: pad + (i/(costHistory.length-1))*(w-pad*2),
      y: h - pad - (v/max)*(h-pad*2)
    }));
    ctx.strokeStyle = '#7c6af7'; ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p,i) => i===0 ? ctx.moveTo(p.x,p.y) : ctx.lineTo(p.x,p.y));
    ctx.stroke();
    const grad = ctx.createLinearGradient(0,0,0,h);
    grad.addColorStop(0,'rgba(124,106,247,0.3)'); grad.addColorStop(1,'rgba(124,106,247,0)');
    ctx.fillStyle = grad; ctx.beginPath();
    points.forEach((p,i) => i===0 ? ctx.moveTo(p.x,p.y) : ctx.lineTo(p.x,p.y));
    ctx.lineTo(points[points.length-1].x,h); ctx.lineTo(points[0].x,h);
    ctx.closePath(); ctx.fill();
    points.forEach(p => { ctx.fillStyle='#7c6af7'; ctx.beginPath(); ctx.arc(p.x,p.y,3,0,Math.PI*2); ctx.fill(); });
  }};
  sparklineChart.draw();
}

function updateSPDGauge(similarity, agreed) {
  if (similarity === null || similarity === undefined) return;
  if (agreed) spdStats.consensus++; else spdStats.escalated++;
  const circle = document.getElementById('spd-gauge-circle');
  const score = document.getElementById('spd-score');
  const status = document.getElementById('spd-status');
  if (circle) {
    circle.style.strokeDashoffset = 314 - (Math.min(similarity,1)*314);
    circle.style.stroke = similarity >= 0.75 ? '#22d3a5' : '#f43f5e';
  }
  if (score) { score.textContent = similarity.toFixed(2); score.style.color = similarity >= 0.75 ? 'var(--green)' : 'var(--red)'; }
  if (status) { status.textContent = agreed ? 'Consensus — frontier model skipped' : 'No consensus — escalated to Llama 70B'; status.style.color = agreed ? 'var(--green)' : 'var(--red)'; }
  const cc = document.getElementById('spd-consensus-count');
  const ec = document.getElementById('spd-escalate-count');
  if (cc) cc.textContent = spdStats.consensus;
  if (ec) ec.textContent = spdStats.escalated;
}

function switchTab(tab, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + tab).classList.add('active');
  el.classList.add('active');
}

async function sendRequest() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) return;
  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="thinking">consulting models</span>...';
  document.getElementById('result-box').classList.remove('show');
  document.getElementById('prompt-input').value = '';
  const strategy = document.getElementById('strategy-select').value;
  const t0 = Date.now();
  try {
    const res = await fetch('/v1/stream', {
      method: 'POST',
      headers: {'Content-Type':'application/json','X-API-Key':API_KEY,'X-Team-ID':'engineering'},
      body: JSON.stringify({prompt, strategy, max_tokens: 500})
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '', meta = null, fullContent = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.type === 'meta') {
            meta = ev;
            const lat = Date.now() - t0;
            document.getElementById('result-box').classList.add('show');
            document.getElementById('result-content').innerHTML = '';
            document.getElementById('routing-explain').textContent = '-> ' + (ev.routing_reason || '');
            document.getElementById('complexity-fill').style.width = '0%';
            document.getElementById('complexity-val').textContent = '';
            const baselineCost = (ev.tokens||100)*GPT4O_COST_PER_TOKEN;
            const saved = baselineCost - (ev.cost||0);
            const savedPct = baselineCost>0 ? Math.round(saved/baselineCost*100) : 0;
            const compTag = ev.compression_applied
              ? '<span class="tag tag-compressed">compressed ' + Math.round((1-(ev.compression_ratio||1))*100) + '% fewer tokens</span>'
              : '<span class="tag" style="background:rgba(107,107,138,.1);color:var(--muted)">no compression</span>';
            const spdTag = ev.spd_similarity
              ? '<span class="tag" style="background:rgba(56,189,248,.15);color:var(--blue)">SPD ' + ev.spd_similarity + '</span>'
              : '';
            if (ev.provider === 'prompt_dna_cache') {
              document.getElementById('result-meta').innerHTML =
                '<span class="tag" style="background:rgba(124,106,247,.3);color:var(--accent)">DNA CACHE HIT</span>' +
                '<span class="tag tag-cost">$0.000000</span>' +
                '<span class="tag tag-lat">' + lat + 'ms</span>' +
                '<span class="tag" style="background:rgba(124,106,247,.2);color:var(--accent)">100% saved</span>';
            } else {
              document.getElementById('result-meta').innerHTML =
                '<span class="tag tag-model">' + ev.model + '</span>' +
                '<span class="tag tag-cost">$' + (ev.cost||0).toFixed(6) + '</span>' +
                '<span class="tag tag-lat">' + lat + 'ms</span>' +
                '<span class="tag tag-saved">saved ' + savedPct + '% ($' + Math.max(0,saved).toFixed(6) + ') vs GPT-4o</span>' +
                compTag + spdTag;
            }
            btn.innerHTML = '<span class="thinking">streaming</span>...';
          } else if (ev.type === 'token') {
            fullContent += ev.text;
            document.getElementById('result-content').innerHTML = fullContent.replace(/\\n/g, '<br>');
          } else if (ev.type === 'done') {
            if (meta) {
              const lat = Date.now()-t0;
              const baselineCost = (meta.tokens||100)*GPT4O_COST_PER_TOKEN;
              const fd = {
                content:fullContent, model_used:meta.model, provider:meta.provider,
                query_type:meta.query_type||'unknown', complexity_score:0,
                tokens_used:meta.tokens, estimated_cost_usd:meta.cost,
                routing_reason:meta.routing_reason, fallback_used:false,
                compression_applied:meta.compression_applied,
                original_tokens:meta.original_tokens,
                compressed_tokens:meta.compressed_tokens,
                compression_ratio:meta.compression_ratio,
                compression_savings_usd:meta.compression_savings||0
              };
              updateStats(fd, lat, baselineCost);
              addFeedItem(fd, lat, prompt);
              updateHistoryTable(fd, lat, prompt, baselineCost);
              if (meta.spd_similarity !== null && meta.spd_similarity !== undefined) {
                updateSPDGauge(meta.spd_similarity, meta.spd_similarity >= 0.75);
              }
              costHistory.push(meta.cost||0);
              if (costHistory.length>20) costHistory.shift();
              if (sparklineChart) sparklineChart.draw();
            }
          } else if (ev.type === 'error') {
            document.getElementById('result-content').innerHTML = 'Error: ' + ev.message;
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    document.getElementById('result-content').innerHTML = 'Error: ' + e.message;
    document.getElementById('result-box').classList.add('show');
  }
  btn.disabled = false;
  btn.innerHTML = 'Send Request';
}

function updateStats(data, latency, baselineCost) {
  stats.requests++;
  stats.totalCost += data.estimated_cost_usd||0;
  stats.totalBaseline += baselineCost;
  stats.totalLatency += latency;
  stats.success++;
  stats.totalCompressionSaved += data.compression_savings_usd||0;
  stats.totalOriginalTokens += data.original_tokens||0;
  stats.totalCompressedTokens += data.compressed_tokens||0;
  const m = data.model_used;
  stats.models[m] = (stats.models[m]||0)+1;
  document.getElementById('kpi-requests').textContent = stats.requests;
  document.getElementById('kpi-cost').textContent = '$'+stats.totalCost.toFixed(4);
  document.getElementById('kpi-latency').textContent = Math.round(stats.totalLatency/stats.requests)+'ms';
  document.getElementById('kpi-success').textContent = Math.round(stats.success/stats.requests*100)+'%';
  const saved = stats.totalBaseline - stats.totalCost;
  const pct = stats.totalBaseline>0 ? Math.round(saved/stats.totalBaseline*100) : 0;
  document.getElementById('savings-amount').textContent = '$'+Math.max(0,saved).toFixed(4);
  document.getElementById('baseline-cost').textContent = '$'+stats.totalBaseline.toFixed(4);
  document.getElementById('actual-cost').textContent = '$'+stats.totalCost.toFixed(4);
  document.getElementById('savings-pct').textContent = pct+'%';
  document.getElementById('savings-sub').textContent = 'Smart routing saved '+pct+'% vs hypothetical GPT-4o baseline';
  const removed = stats.totalOriginalTokens - stats.totalCompressedTokens;
  const cpct = stats.totalOriginalTokens>0 ? Math.round(removed/stats.totalOriginalTokens*100) : 0;
  document.getElementById('compression-amount').textContent = '$'+stats.totalCompressionSaved.toFixed(6);
  document.getElementById('total-original-tokens').textContent = stats.totalOriginalTokens;
  document.getElementById('total-compressed-tokens').textContent = stats.totalCompressedTokens;
  document.getElementById('compression-pct').textContent = cpct+'%';
  document.getElementById('compression-sub').textContent = removed+' tokens removed by LLMLingua-2 this session';
  updateModelBars();
  updateCacheStats();
}

function updateModelBars() {
  const total = stats.requests;
  const html = Object.entries(stats.models).sort((a,b)=>b[1]-a[1]).map(([model,count]) => {
    const pct = Math.round(count/total*100);
    const col = MODEL_COLORS[model]||'#6b6b8a';
    const short = model.length>22 ? model.slice(0,20)+'..' : model;
    return '<div class="bar-row"><div class="bar-label">'+short+'</div><div class="bar-track"><div class="bar-fill" style="width:'+pct+'%;background:'+col+'"></div></div><div class="bar-val">'+pct+'%</div></div>';
  }).join('');
  document.getElementById('model-bars').innerHTML = html;
}

async function updateCacheStats() {
  try {
    const r = await fetch('/v1/cache/stats', {headers:{'X-API-Key':API_KEY}});
    const d = await r.json();
    if (!d.available) return;
    const hitCost = (d.hits||0)*0.0000025*100;
    document.getElementById('cache-hits').textContent = d.hits||0;
    document.getElementById('cache-hit-rate').textContent = (d.hit_rate_pct||0)+'%';
    document.getElementById('cache-size').textContent = d.cache_size||0;
    document.getElementById('cache-saved').textContent = '$'+hitCost.toFixed(6);
    if (d.hits>0) document.getElementById('cache-sub').textContent = d.hits+' requests served from semantic cache at zero API cost';
  } catch(e) {}
}

function addFeedItem(data, latency, prompt) {
  const feed = document.getElementById('feed');
  if (feed.querySelector('.empty-state')) feed.innerHTML = '';
  const cb = data.compression_applied
    ? '<span style="font-size:10px;color:var(--blue);margin-left:4px">'+data.original_tokens+'->'+data.compressed_tokens+'</span>'
    : '';
  const item = document.createElement('div');
  item.className = 'feed-item';
  item.innerHTML =
    '<div class="feed-dot" style="background:'+(MODEL_COLORS[data.model_used]||'#6b6b8a')+'"></div>'+
    '<div class="feed-model">'+data.model_used+cb+'</div>'+
    '<div class="feed-type">'+data.query_type+'</div>'+
    '<div style="font-size:11px;color:var(--muted);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+prompt.slice(0,45)+'...</div>'+
    '<div class="feed-lat">'+latency+'ms</div>'+
    '<div class="feed-cost">$'+(data.estimated_cost_usd||0).toFixed(5)+'</div>';
  feed.insertBefore(item, feed.firstChild);
  if (feed.children.length>20) feed.removeChild(feed.lastChild);
}

function updateHistoryTable(data, latency, prompt, baselineCost) {
  const table = document.getElementById('history-table');
  if (table.querySelector('.empty-state')) {
    table.innerHTML = '<table><thead><tr><th>Prompt</th><th>Model</th><th>Type</th><th style="text-align:right">Latency</th><th style="text-align:right">Cost</th><th style="text-align:right">Routing saved</th><th style="text-align:right">Tokens</th><th style="text-align:right">Compressed</th></tr></thead><tbody id="history-body"></tbody></table>';
  }
  const saved = baselineCost-(data.estimated_cost_usd||0);
  const savedPct = baselineCost>0 ? Math.round(saved/baselineCost*100) : 0;
  const compStr = data.compression_applied ? data.original_tokens+'->'+data.compressed_tokens+' ('+Math.round((1-(data.compression_ratio||1))*100)+'%)' : 'none';
  const typeColors = {coding:'var(--accent)',reasoning:'var(--blue)',summarization:'var(--green)',creative:'var(--amber)',simple:'var(--muted)',unknown:'var(--muted)'};
  const row = document.createElement('tr');
  row.innerHTML =
    '<td style="color:var(--text);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+prompt.slice(0,60)+(prompt.length>60?'...':'')+'</td>'+
    '<td style="color:var(--accent)">'+data.model_used+'</td>'+
    '<td style="color:'+(typeColors[data.query_type]||'var(--muted)')+'">'+data.query_type+'</td>'+
    '<td style="text-align:right;color:var(--muted)">'+latency+'ms</td>'+
    '<td style="text-align:right;color:var(--amber)">$'+(data.estimated_cost_usd||0).toFixed(6)+'</td>'+
    '<td style="text-align:right;color:var(--green)">'+savedPct+'% ($'+Math.max(0,saved).toFixed(6)+')</td>'+
    '<td style="text-align:right;color:var(--muted)">'+(data.tokens_used||0)+'</td>'+
    '<td style="text-align:right;color:var(--blue)">'+compStr+'</td>';
  const tbody = document.getElementById('history-body');
  if (tbody) tbody.insertBefore(row, tbody.firstChild);
}

async function runComparison() {
  const prompt = document.getElementById('compare-input').value.trim();
  if (!prompt) return;
  const btn = document.getElementById('compare-btn');
  btn.disabled = true;
  btn.innerHTML = 'querying all providers...';
  document.getElementById('compare-results').innerHTML = '<div class="loading-state"><span class="spin">|</span> Sending to Groq 8B, Gemini 2.5 Flash, and Groq 70B...</div>';
  const models = [
    {model:'llama-3.1-8b-instant', strategy:'performance', label:'Groq Llama 8B', color:'#7c6af7'},
    {model:'gemini-2.5-flash', strategy:'performance', label:'Gemini 2.5 Flash', color:'#22d3a5'},
    {model:'llama-3.3-70b-versatile', strategy:'performance', label:'Groq Llama 70B', color:'#38bdf8'},
  ];
  const results = await Promise.allSettled(models.map(async m => {
    const t0 = Date.now();
    const res = await fetch('/v1/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json','X-API-Key':API_KEY,'X-Team-ID':'engineering'},
      body:JSON.stringify({prompt, strategy:m.strategy, max_tokens:400, preferred_model:m.model})
    });
    const data = await res.json();
    return {...data, latency:Date.now()-t0, label:m.label, color:m.color};
  }));
  const fulfilled = results.filter(r=>r.status==='fulfilled');
  const fastest = fulfilled.length>0 ? fulfilled.reduce((a,b)=>a.value.latency<b.value.latency?a:b) : null;
  const html = results.map(r => {
    if (r.status==='rejected') return '<div class="compare-card"><div style="color:var(--red)">Request failed</div></div>';
    const d = r.value;
    const isWinner = fastest && d.latency===fastest.value.latency;
    return '<div class="compare-card" style="border-color:'+(isWinner?d.color:'var(--border)')+'">'+
      '<div class="compare-header"><span class="compare-model" style="color:'+d.color+'">'+d.label+'</span>'+(isWinner?'<span class="winner-badge">FASTEST</span>':'')+' </div>'+
      '<div class="compare-content">'+(d.content||'No response')+'</div>'+
      '<div class="compare-footer"><span style="color:'+d.color+'">'+d.latency+'ms</span><span style="color:var(--amber)">$'+(d.estimated_cost_usd||0).toFixed(6)+'</span><span style="color:var(--muted)">'+(d.tokens_used||0)+' tokens</span></div></div>';
  }).join('');
  document.getElementById('compare-results').innerHTML = '<div class="compare-grid">'+html+'</div>';
  btn.disabled = false;
  btn.innerHTML = 'Compare All Providers';
}

function clearDemo() {
  document.getElementById('prompt-input').value = '';
  document.getElementById('result-box').classList.remove('show');
}

async function loadUserNav() {
  try {
    const token = localStorage.getItem('token');
    const h = token
      ? {'X-API-Key': API_KEY, 'Authorization': 'Bearer ' + token}
      : {'X-API-Key': API_KEY};
    const r = await fetch('/auth/me', {headers: h});
    if (r.ok) {
      const u = await r.json();
      if (!u.verified) {
        const header = document.querySelector('header');
        if (header) {
          const banner = document.createElement('div');
          banner.style.cssText = 'background:rgba(245,158,11,.1);border-bottom:1px solid rgba(245,158,11,.3);padding:10px 40px;font-size:13px;color:var(--amber);display:flex;align-items:center;justify-content:space-between;position:relative;z-index:200';
          banner.innerHTML = '<span>Please verify your email address.</span><button onclick="resendVerification()" style="background:rgba(245,158,11,.2);border:1px solid rgba(245,158,11,.4);color:var(--amber);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px">Resend email</button>';
          header.insertAdjacentElement('afterend', banner);
        }
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
    alert('Verification email sent.');
  } catch(e) {}
}

window.addEventListener('load', () => {
  setTimeout(initSparkline, 100);
  loadUserNav();
});
window.addEventListener('resize', () => { if (sparklineChart) sparklineChart.draw(); });
document.getElementById('prompt-input').addEventListener('keydown', e => {
  if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendRequest(); }
});
</script>"""

with open('gateway/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_before + clean_script + html_after)
print('Done')