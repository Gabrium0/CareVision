const $ = id => document.getElementById(id);
const state = {csrf:'', summary:null, selectedCase:null};
const svgNS = 'http://www.w3.org/2000/svg';

function node(tag, text, cls) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = String(text);
  if (cls) item.className = cls;
  return item;
}
function formatTime(value) { return value ? new Date(value * 1000).toLocaleString() : '—'; }
function selection() { return `subject_id=${encodeURIComponent($('subject').value)}&range=${$('range').value}`; }

function renderCounts(counts) {
  const root = $('counts'); root.replaceChildren();
  for (const [key,value] of Object.entries(counts || {})) {
    const card=node('div',undefined,'metric'); card.append(node('strong',value),node('span',key.replaceAll('_',' '))); root.append(card);
  }
}
function renderCases(cases) {
  const root=$('cases'); root.replaceChildren(); $('case-count').textContent=`${cases.length} matching cases`;
  if (!cases.length) { root.append(node('div','No matching cases.','empty')); return; }
  for (const item of cases) {
    const card=node('article',undefined,'case');
    const copy=node('div'); copy.append(node('h3',`${item.module}.${item.key}`),node('p',item.summary),node('p',`${item.subject_id} · opened ${formatTime(item.opened_at)} · signal ${item.signal_active?'active':'cleared'}`));
    const badge=node('span',item.status,`badge ${item.status}`); const button=node('button','Review'); button.addEventListener('click',()=>openCase(item.id));
    card.append(copy,badge,button); root.append(card);
  }
}
function chart(series) {
  const card=node('article',undefined,'chart'); card.append(node('h3',`${series.subject_id} · ${series.module}.${series.key}`));
  const values=series.points || []; if (!values.length) return card;
  const svg=document.createElementNS(svgNS,'svg'); svg.setAttribute('viewBox','0 0 600 150'); svg.setAttribute('role','img');
  const lows=values.map(p=>p.min), highs=values.map(p=>p.max); const lo=Math.min(...lows), hi=Math.max(...highs); const span=Math.max(.0001,hi-lo);
  const xy=(p,i)=>[values.length===1?300:i*600/(values.length-1),140-(p.mean-lo)*125/span];
  const band=document.createElementNS(svgNS,'polygon'); const upper=values.map((p,i)=>`${xy({...p,mean:p.max},i).join(',')}`); const lower=values.map((p,i)=>`${xy({...p,mean:p.min},i).join(',')}`).reverse(); band.setAttribute('points',[...upper,...lower].join(' ')); band.setAttribute('fill','#72e0af22');
  const line=document.createElementNS(svgNS,'polyline'); line.setAttribute('points',values.map((p,i)=>xy(p,i).join(',')).join(' ')); line.setAttribute('fill','none'); line.setAttribute('stroke','#72e0af'); line.setAttribute('stroke-width','3'); svg.append(band,line); card.append(svg);
  const table=node('table'); const last=values.at(-1); const row=node('tr'); row.append(node('td',`Latest bucket (${last.count} samples)`),node('td',`${last.mean.toFixed(2)} · ${last.min.toFixed(2)}–${last.max.toFixed(2)}`)); table.append(row); card.append(table); return card;
}
function renderTrends(series) { const root=$('trends'); root.replaceChildren(); const visible=(series||[]).filter(s=>s.points?.length); if(!visible.length)root.append(node('div','No numeric trend samples in this period.','empty')); else visible.forEach(s=>root.append(chart(s))); }
function renderRetention(r) { $('retention').textContent=`${r.event_rows} events (${r.expired_event_rows} expired), ${r.case_rows} cases (${r.unresolved_case_rows} unresolved), ${r.history_rows} numeric samples (${r.expired_history_rows} expired). Policies: events ${r.event_retention_days}d, resolved cases ${r.resolved_case_retention_days}d, history ${r.history_retention_days}d.`; }

async function refresh() {
  $('connection').textContent='Refreshing…';
  try {
    const response=await fetch(`/caregiver/api/summary?${selection()}&status=${$('case-status').value}`,{cache:'no-store'}); if(!response.ok)throw new Error(`HTTP ${response.status}`);
    state.summary=await response.json(); renderCounts(state.summary.counts); renderCases(state.summary.cases); renderTrends(state.summary.series); renderRetention(state.summary.retention);
    $('json-export').href=`/caregiver/api/export?${selection()}&format=json`; $('csv-export').href=`/caregiver/api/export?${selection()}&format=csv`; $('connection').textContent=`Updated ${new Date().toLocaleTimeString()}`;
  } catch(error) { $('connection').textContent=`Unavailable: ${error.message}`; }
}
function renderCaseDetail(item) {
  state.selectedCase=item; $('dialog-title').textContent=`${item.module}.${item.key}`; const root=$('case-detail'); root.replaceChildren(); root.append(node('p',item.summary),node('p',`${item.subject_id} · ${item.status} · signal ${item.signal_active?'active':'cleared'} · version ${item.version}`));
  const audit=node('div',undefined,'audit'); for(const action of item.actions||[]){const row=node('div'); row.append(node('strong',action.action),node('time',` ${formatTime(action.timestamp)}${action.channel?` · ${action.channel}`:''}`)); if(action.note)row.append(node('p',action.note)); audit.append(row);} root.append(audit);
  $('ack').disabled=item.status!=='open'; $('resolve').disabled=item.status==='resolved'; $('case-note').value='';
}
async function openCase(id) { const response=await fetch(`/caregiver/api/cases/${encodeURIComponent(id)}`,{cache:'no-store'}); if(!response.ok)return; renderCaseDetail(await response.json()); $('case-dialog').showModal(); }
async function mutate(action) {
  if(!state.selectedCase)return; const note=$('case-note').value.trim(); const body={version:state.selectedCase.version}; if(note)body.note=note;
  const response=await fetch(`/caregiver/api/cases/${state.selectedCase.id}/${action}`,{method:'POST',headers:{'Content-Type':'application/json','X-Caregiver-CSRF':state.csrf},body:JSON.stringify(body)});
  const result=await response.json(); if(!response.ok){alert(result.error||`HTTP ${response.status}`); if(response.status===409)openCase(state.selectedCase.id); return;} renderCaseDetail(result.case); await refresh();
}
async function purge() { if(!confirm('Permanently purge only data whose retention period has expired?'))return; const response=await fetch('/caregiver/api/retention/purge-expired',{method:'POST',headers:{'Content-Type':'application/json','X-Caregiver-CSRF':state.csrf},body:JSON.stringify({confirm:'purge-expired'})}); if(response.ok)await refresh(); }
async function start() {
  const response=await fetch('/caregiver/api/bootstrap',{cache:'no-store'}); const boot=await response.json(); state.csrf=boot.csrf_token; const select=$('subject'); select.replaceChildren(); for(const subject of boot.subjects){const option=node('option',subject); option.value=subject; select.append(option);} await refresh();
}
for(const id of ['subject','range','case-status'])$(id).addEventListener('change',refresh); $('refresh').addEventListener('click',refresh); $('add-note').addEventListener('click',()=>mutate('note')); $('ack').addEventListener('click',()=>mutate('acknowledge')); $('resolve').addEventListener('click',()=>mutate('resolve')); $('purge').addEventListener('click',purge); start();
