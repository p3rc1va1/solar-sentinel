/* ============================================================
   Solar Sentinel — Dashboard JavaScript
   ============================================================ */

const API = window.location.origin;

// ── Navigation ────────────────────────────────────────────────

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const page = link.dataset.page;

        // Update nav
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');

        // Show page
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(`page-${page}`).classList.add('active');

        // Page-specific init
        if (page === 'dashboard') loadDashboard();
        if (page === 'detections') loadDetections();
        if (page === 'reports') loadReports();
        if (page === 'live') startLiveFeed();
        if (page === 'settings') loadSettings();
    });
});

// ── API Helpers ───────────────────────────────────────────────

async function fetchJSON(endpoint) {
    try {
        const resp = await fetch(`${API}${endpoint}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`Fetch error: ${endpoint}`, err);
        return null;
    }
}

async function putJSON(endpoint, data) {
    const resp = await fetch(`${API}${endpoint}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    return resp.json();
}

function formatTime(isoStr) {
    if (!isoStr) return '—';
    const d = new Date(isoStr);
    return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });
}

function severityBadge(severity) {
    const cls = {
        'CRITICAL': 'badge-critical',
        'WARNING': 'badge-warning',
        'INFO': 'badge-info',
    }[severity] || 'badge-info';
    return `<span class="badge ${cls}">${severity}</span>`;
}

function confidenceBadge(conf) {
    const pct = (conf * 100).toFixed(0);
    if (conf >= 0.7) return `<span class="badge badge-critical">${pct}%</span>`;
    if (conf >= 0.45) return `<span class="badge badge-warning">${pct}%</span>`;
    return `<span class="badge badge-info">${pct}%</span>`;
}

// ── Dashboard ─────────────────────────────────────────────────

async function loadDashboard() {
    // Health
    const health = await fetchJSON('/health');
    if (health) {
        const sys = health.system;
        const dot = document.querySelector('.status-dot');
        const txt = document.querySelector('.status-text');
        dot.classList.add('online');
        txt.textContent = 'Online';

        document.getElementById('statTemp').textContent =
            sys.cpu_temp_c !== null ? `${sys.cpu_temp_c.toFixed(1)}°C` : 'N/A';
        document.getElementById('statMemory').textContent = `${sys.memory_used_percent}%`;
        document.getElementById('statMemoryDetail').textContent =
            `${sys.memory_available_mb} MB available`;
        document.getElementById('statDisk').textContent = `${sys.disk_used_percent}%`;
        document.getElementById('statDiskDetail').textContent =
            `${sys.disk_free_gb} GB free`;

        // API usage
        const usage = health.gemini_usage_today || [];
        const total = usage.reduce((s, u) => s + (u.count || 0), 0);
        document.getElementById('statApi').textContent = total;
    } else {
        document.querySelector('.status-dot').classList.add('error');
        document.querySelector('.status-text').textContent = 'Offline';
    }

    // Recent detections
    const detData = await fetchJSON('/detections?limit=5');
    const detTbody = document.querySelector('#dashDetections tbody');
    detTbody.innerHTML = '';
    if (detData?.detections) {
        detData.detections.forEach(d => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${formatTime(d.timestamp)}</td>
                <td>${d.defect_class}</td>
                <td>${confidenceBadge(d.confidence)}</td>
                <td>${d.report_id ? '<span class="badge badge-success">Reported</span>' : '—'}</td>
            `;
            detTbody.appendChild(row);
        });
    }

    // Recent reports
    const repData = await fetchJSON('/reports?limit=5');
    const repTbody = document.querySelector('#dashReports tbody');
    repTbody.innerHTML = '';
    if (repData?.reports) {
        repData.reports.forEach(r => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${formatTime(r.created_at)}</td>
                <td>${severityBadge(r.severity)}</td>
                <td>${r.qa_approved ? '<span class="badge badge-success">Approved</span>' : '<span class="badge badge-warning">Pending</span>'}</td>
                <td>${r.qa_score}/10</td>
            `;
            repTbody.appendChild(row);
        });
    }
}

// ── Detections ────────────────────────────────────────────────

async function loadDetections() {
    const data = await fetchJSON('/detections?limit=100');
    const tbody = document.querySelector('#detectionsTable tbody');
    tbody.innerHTML = '';
    if (data?.detections) {
        data.detections.forEach(d => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${d.id}</td>
                <td>${formatTime(d.timestamp)}</td>
                <td>${d.defect_class}</td>
                <td>${confidenceBadge(d.confidence)}</td>
                <td>${d.panel_id}</td>
                <td>${d.report_id || '—'}</td>
            `;
            tbody.appendChild(row);
        });
    }
}

// ── Reports ───────────────────────────────────────────────────

async function loadReports() {
    const data = await fetchJSON('/reports?limit=100');
    const tbody = document.querySelector('#reportsTable tbody');
    tbody.innerHTML = '';
    if (data?.reports) {
        data.reports.forEach(r => {
            const row = document.createElement('tr');
            row.style.cursor = 'pointer';
            row.innerHTML = `
                <td>${r.id}</td>
                <td>${formatTime(r.created_at)}</td>
                <td>${severityBadge(r.severity)}</td>
                <td>${r.urgency}</td>
                <td>${r.qa_score}/10</td>
                <td>${r.qa_approved ? '<span class="badge badge-success">Yes</span>' : '<span class="badge badge-warning">No</span>'}</td>
            `;
            row.addEventListener('click', () => showReportDetail(r.id));
            tbody.appendChild(row);
        });
    }
}

async function showReportDetail(id) {
    const data = await fetchJSON(`/reports/${id}`);
    if (data) {
        document.getElementById('reportDetail').style.display = 'block';
        document.getElementById('reportContent').textContent = data.report_markdown || 'No content';
    }
}

// ── Live Feed ─────────────────────────────────────────────────

function startLiveFeed() {
    const img = document.getElementById('liveFeed');
    img.src = `${API}/camera/feed`;
}

// ── Settings ──────────────────────────────────────────────────

async function loadSettings() {
    const data = await fetchJSON('/settings');
    if (!data) return;

    // Gemini
    document.getElementById('geminiKey').value = data.gemini?.gemini_api_key || '';

    // Email
    const n = data.notifications || {};
    document.getElementById('emailEnabled').checked = n.email_enabled || false;
    document.getElementById('emailAddr').value = n.email_address || '';
    document.getElementById('smtpHost').value = n.smtp_host || 'smtp.gmail.com';
    document.getElementById('smtpPort').value = n.smtp_port || 587;
    document.getElementById('smtpUser').value = n.smtp_username || '';
    document.getElementById('smtpPass').value = n.smtp_password || '';

    // Telegram
    document.getElementById('telegramEnabled').checked = n.telegram_enabled || false;
    document.getElementById('tgToken').value = n.telegram_bot_token || '';
    document.getElementById('tgChat').value = n.telegram_chat_id || '';

    // Detection
    const d = data.detection || {};
    document.getElementById('confHigh').value = d.confidence_high || 0.70;
    document.getElementById('confHighVal').textContent = d.confidence_high || 0.70;
    document.getElementById('confMedium').value = d.confidence_medium || 0.45;
    document.getElementById('confMediumVal').textContent = d.confidence_medium || 0.45;
}

// Range slider live values
document.getElementById('confHigh').addEventListener('input', (e) => {
    document.getElementById('confHighVal').textContent = e.target.value;
});
document.getElementById('confMedium').addEventListener('input', (e) => {
    document.getElementById('confMediumVal').textContent = e.target.value;
});

// Save settings
document.getElementById('btnSaveSettings').addEventListener('click', async () => {
    const payload = {
        gemini: {
            gemini_api_key: document.getElementById('geminiKey').value,
        },
        notifications: {
            email_enabled: document.getElementById('emailEnabled').checked,
            email_address: document.getElementById('emailAddr').value,
            smtp_host: document.getElementById('smtpHost').value,
            smtp_port: parseInt(document.getElementById('smtpPort').value),
            smtp_username: document.getElementById('smtpUser').value,
            smtp_password: document.getElementById('smtpPass').value,
            telegram_enabled: document.getElementById('telegramEnabled').checked,
            telegram_bot_token: document.getElementById('tgToken').value,
            telegram_chat_id: document.getElementById('tgChat').value,
        },
        detection: {
            confidence_high: parseFloat(document.getElementById('confHigh').value),
            confidence_medium: parseFloat(document.getElementById('confMedium').value),
            capture_interval_minutes: 15,
            capture_interval_after_high: 5,
            capture_interval_after_clean: 30,
        },
    };

    try {
        await putJSON('/settings', payload);
        document.getElementById('saveStatus').textContent = '✓ Saved';
        setTimeout(() => {
            document.getElementById('saveStatus').textContent = '';
        }, 3000);
    } catch (err) {
        document.getElementById('saveStatus').textContent = '✗ Error saving';
        document.getElementById('saveStatus').style.color = 'var(--danger)';
    }
});

// Manual capture button
document.getElementById('btnCapture').addEventListener('click', async () => {
    const btn = document.getElementById('btnCapture');
    btn.disabled = true;
    btn.textContent = 'Capturing...';
    await fetch(`${API}/camera/capture`, { method: 'POST' });
    btn.textContent = 'Capture Now';
    btn.disabled = false;
    setTimeout(loadDashboard, 2000);
});

// ── Init ──────────────────────────────────────────────────────

loadDashboard();
// Auto-refresh dashboard every 60 seconds
setInterval(() => {
    if (document.querySelector('#page-dashboard.active')) {
        loadDashboard();
    }
}, 60000);
