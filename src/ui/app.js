/* ============================================================
   Solar Sentinel — Dashboard JavaScript
   ============================================================ */

const API = window.location.origin;

// ── Toast Notifications ───────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

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

function formatTimeRelative(isoStr) {
    if (!isoStr) return '—';
    const d = new Date(isoStr);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return 'Just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return formatTime(isoStr);
}

function severityBadge(severity) {
    const cls = {
        'CRITICAL': 'badge-critical',
        'WARNING': 'badge-warning',
        'INFO': 'badge-info',
    }[severity] || 'badge-info';
    return `<span class="badge ${cls}">${severity}</span>`;
}

function classBadge(className) {
    const cls = {
        'damage': 'badge-damage',
        'blockage': 'badge-blockage',
        'healthy': 'badge-healthy',
    }[className] || 'badge-info';
    return `<span class="badge ${cls}">${className}</span>`;
}

function confidenceBadge(conf) {
    const pct = (conf * 100).toFixed(0);
    if (conf >= 0.7) return `<span class="badge badge-critical">${pct}%</span>`;
    if (conf >= 0.45) return `<span class="badge badge-warning">${pct}%</span>`;
    return `<span class="badge badge-info">${pct}%</span>`;
}

// ── Charts (Chart.js instances) ───────────────────────────────

let trendChart = null;
let classChart = null;

function initCharts(detections) {
    // Detection trend — count detections per day for the last 7 days
    const now = new Date();
    const days = [];
    const dayCounts = {};

    for (let i = 6; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        const key = d.toISOString().slice(0, 10);
        days.push(key);
        dayCounts[key] = { damage: 0, blockage: 0, healthy: 0 };
    }

    (detections || []).forEach(d => {
        if (!d.timestamp) return;
        const key = new Date(d.timestamp).toISOString().slice(0, 10);
        if (dayCounts[key] && dayCounts[key][d.defect_class] !== undefined) {
            dayCounts[key][d.defect_class]++;
        }
    });

    const labels = days.map(d => {
        const dt = new Date(d + 'T00:00:00');
        return dt.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
    });

    // Trend chart
    const trendCtx = document.getElementById('chartTrend');
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(trendCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Damage',
                    data: days.map(d => dayCounts[d].damage),
                    backgroundColor: 'rgba(239, 68, 68, 0.7)',
                    borderRadius: 4,
                },
                {
                    label: 'Blockage',
                    data: days.map(d => dayCounts[d].blockage),
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderRadius: 4,
                },
                {
                    label: 'Healthy',
                    data: days.map(d => dayCounts[d].healthy),
                    backgroundColor: 'rgba(34, 197, 94, 0.7)',
                    borderRadius: 4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9aa0b0', font: { family: "'Inter', sans-serif", size: 11 } }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    ticks: { color: '#6b7280', font: { size: 10 } },
                    grid: { color: 'rgba(42, 52, 80, 0.5)' },
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    ticks: { color: '#6b7280', stepSize: 1 },
                    grid: { color: 'rgba(42, 52, 80, 0.5)' },
                },
            },
        },
    });

    // Class distribution donut
    const classCounts = { damage: 0, blockage: 0, healthy: 0 };
    (detections || []).forEach(d => {
        if (classCounts[d.defect_class] !== undefined) classCounts[d.defect_class]++;
    });

    const classCtx = document.getElementById('chartClasses');
    if (classChart) classChart.destroy();
    classChart = new Chart(classCtx, {
        type: 'doughnut',
        data: {
            labels: ['Damage', 'Blockage', 'Healthy'],
            datasets: [{
                data: [classCounts.damage, classCounts.blockage, classCounts.healthy],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(34, 197, 94, 0.8)',
                ],
                borderColor: '#1a2035',
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#9aa0b0', font: { family: "'Inter', sans-serif", size: 11 }, padding: 12 }
                }
            },
        },
    });
}

// ── Detection Detail Modal ────────────────────────────────────

const detModal = document.getElementById('detectionModal');
const modalClose = document.getElementById('modalClose');

modalClose.addEventListener('click', () => detModal.classList.remove('visible'));
detModal.addEventListener('click', (e) => {
    if (e.target === detModal) detModal.classList.remove('visible');
});

function showDetectionDetail(det) {
    document.getElementById('modalTitle').textContent = `Detection #${det.id}`;
    document.getElementById('modalClass').innerHTML = classBadge(det.defect_class);
    document.getElementById('modalConf').innerHTML = confidenceBadge(det.confidence);
    document.getElementById('modalPanel').textContent = det.panel_id || '—';
    document.getElementById('modalTime').textContent = formatTime(det.timestamp);

    const bbox = det.bbox || {};
    document.getElementById('modalBbox').textContent = bbox.x1 !== undefined
        ? `(${Math.round(bbox.x1)}, ${Math.round(bbox.y1)}) → (${Math.round(bbox.x2)}, ${Math.round(bbox.y2)})`
        : '—';

    document.getElementById('modalReport').innerHTML = det.report_id
        ? `<a href="#" onclick="event.preventDefault(); detModal.classList.remove('visible'); navigateTo('reports'); setTimeout(() => showReportDetail(${det.report_id}), 300);">View Report #${det.report_id}</a>`
        : '<span style="color:var(--text-muted)">None</span>';

    // Image — try to load from API
    const img = document.getElementById('modalImage');
    if (det.image_path && det.image_path !== 'data/demo/placeholder.jpg') {
        const filename = det.image_path.split('/').pop();
        img.src = `${API}/images/${filename}`;
        img.alt = `Detection ${det.id}`;
        img.onerror = () => { img.src = ''; img.alt = 'Image not available'; };
    } else {
        img.src = '';
        img.alt = 'No image available';
    }

    detModal.classList.add('visible');
}

function navigateTo(page) {
    document.querySelectorAll('.nav-link').forEach(l => {
        l.classList.toggle('active', l.dataset.page === page);
    });
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
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
        dot.classList.remove('error');
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
        document.querySelector('.status-dot').classList.remove('online');
        document.querySelector('.status-text').textContent = 'Offline';
    }

    // Sensor data
    const sensor = await fetchJSON('/sensor');
    if (sensor && sensor.available) {
        document.getElementById('statSensor').textContent = `${sensor.temperature}°C`;
        document.getElementById('statHumidity').textContent = `${sensor.humidity}% humidity`;
    } else {
        document.getElementById('statSensor').textContent = 'N/A';
        document.getElementById('statHumidity').textContent = 'Sensor unavailable';
    }

    // Recent detections (fetch more for charts)
    const detData = await fetchJSON('/detections?limit=200');
    const detTbody = document.querySelector('#dashDetections tbody');
    detTbody.innerHTML = '';
    if (detData?.detections) {
        // Last capture info
        if (detData.detections.length > 0) {
            const last = detData.detections[0];
            document.getElementById('statLastCapture').textContent = formatTimeRelative(last.timestamp);
            document.getElementById('statLastClass').innerHTML = classBadge(last.defect_class);
        }

        // Charts
        initCharts(detData.detections);

        // Table (top 5)
        detData.detections.slice(0, 5).forEach(d => {
            const row = document.createElement('tr');
            row.className = 'clickable';
            row.innerHTML = `
                <td>${formatTime(d.timestamp)}</td>
                <td>${classBadge(d.defect_class)}</td>
                <td>${confidenceBadge(d.confidence)}</td>
                <td>${d.report_id ? '<span class="badge badge-success">Reported</span>' : '—'}</td>
            `;
            row.addEventListener('click', () => showDetectionDetail(d));
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
            row.className = 'clickable';
            row.innerHTML = `
                <td>${formatTime(r.created_at)}</td>
                <td>${severityBadge(r.severity)}</td>
                <td>${r.qa_approved ? '<span class="badge badge-success">Approved</span>' : '<span class="badge badge-warning">Pending</span>'}</td>
                <td>${r.qa_score}/10</td>
            `;
            row.addEventListener('click', () => {
                navigateTo('reports');
                setTimeout(() => showReportDetail(r.id), 200);
            });
            repTbody.appendChild(row);
        });
    }
}

// ── Detections (with pagination) ──────────────────────────────

let detectionsPage = 0;
const DETECTIONS_PER_PAGE = 20;

async function loadDetections(page = 0) {
    detectionsPage = page;
    const offset = page * DETECTIONS_PER_PAGE;
    const data = await fetchJSON(`/detections?limit=${DETECTIONS_PER_PAGE}&offset=${offset}`);
    const tbody = document.querySelector('#detectionsTable tbody');
    tbody.innerHTML = '';
    if (data?.detections) {
        data.detections.forEach(d => {
            const row = document.createElement('tr');
            row.className = 'clickable';
            row.innerHTML = `
                <td>${d.id}</td>
                <td>${formatTime(d.timestamp)}</td>
                <td>${classBadge(d.defect_class)}</td>
                <td>${confidenceBadge(d.confidence)}</td>
                <td>${d.panel_id}</td>
                <td>${d.report_id || '—'}</td>
            `;
            row.addEventListener('click', () => showDetectionDetail(d));
            tbody.appendChild(row);
        });

        // Pagination
        renderPagination(
            'detectionsPagination',
            page,
            data.count >= DETECTIONS_PER_PAGE,
            (p) => loadDetections(p)
        );
    }
}

// ── Reports (with pagination) ─────────────────────────────────

let reportsPage = 0;
const REPORTS_PER_PAGE = 20;

async function loadReports(page = 0) {
    reportsPage = page;
    const offset = page * REPORTS_PER_PAGE;
    const data = await fetchJSON(`/reports?limit=${REPORTS_PER_PAGE}&offset=${offset}`);
    const tbody = document.querySelector('#reportsTable tbody');
    tbody.innerHTML = '';
    if (data?.reports) {
        data.reports.forEach(r => {
            const row = document.createElement('tr');
            row.className = 'clickable';
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

        renderPagination(
            'reportsPagination',
            page,
            data.count >= REPORTS_PER_PAGE,
            (p) => loadReports(p)
        );
    }
}

async function showReportDetail(id) {
    const data = await fetchJSON(`/reports/${id}`);
    if (data) {
        document.getElementById('reportDetail').style.display = 'block';
        const content = document.getElementById('reportContent');
        // Use marked.js for markdown rendering if available
        if (window.marked) {
            content.innerHTML = marked.parse(data.report_markdown || 'No content');
        } else {
            content.textContent = data.report_markdown || 'No content';
        }
        // Scroll to report detail
        document.getElementById('reportDetail').scrollIntoView({ behavior: 'smooth' });
    }
}

// ── Pagination Helper ─────────────────────────────────────────

function renderPagination(containerId, currentPage, hasMore, onPageChange) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (currentPage === 0 && !hasMore) return; // Single page, no pagination needed

    const prevBtn = document.createElement('button');
    prevBtn.textContent = '← Prev';
    prevBtn.disabled = currentPage === 0;
    prevBtn.addEventListener('click', () => onPageChange(currentPage - 1));
    container.appendChild(prevBtn);

    const info = document.createElement('span');
    info.className = 'page-info';
    info.textContent = `Page ${currentPage + 1}`;
    container.appendChild(info);

    const nextBtn = document.createElement('button');
    nextBtn.textContent = 'Next →';
    nextBtn.disabled = !hasMore;
    nextBtn.addEventListener('click', () => onPageChange(currentPage + 1));
    container.appendChild(nextBtn);
}

// ── Live Feed ─────────────────────────────────────────────────

function startLiveFeed() {
    const img = document.getElementById('liveFeed');
    const overlay = document.getElementById('overlayToggle').checked;
    img.src = `${API}/camera/feed${overlay ? '?overlay=true' : ''}`;
}

document.getElementById('overlayToggle').addEventListener('change', () => {
    if (document.querySelector('#page-live.active')) {
        startLiveFeed();
    }
});

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
        showToast('Settings saved successfully', 'success');
        document.getElementById('saveStatus').textContent = '';
    } catch (err) {
        showToast('Failed to save settings', 'error');
    }
});

// Manual capture button
document.getElementById('btnCapture').addEventListener('click', async () => {
    const btn = document.getElementById('btnCapture');
    btn.disabled = true;
    btn.textContent = 'Capturing...';

    try {
        const resp = await fetch(`${API}/camera/capture`, { method: 'POST' });
        const data = await resp.json();

        if (resp.ok) {
            const count = data.count || 0;
            showToast(`Capture complete — ${count} detection${count !== 1 ? 's' : ''} found`, 'success');
        } else if (resp.status === 429) {
            showToast(data.detail || 'Rate limited. Wait a moment.', 'info');
        } else {
            showToast('Capture failed', 'error');
        }
    } catch (err) {
        showToast('Capture request failed', 'error');
    }

    btn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
        Capture Now
    `;
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
