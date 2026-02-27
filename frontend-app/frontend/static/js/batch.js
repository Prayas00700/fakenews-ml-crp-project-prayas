let currentResults = null;

// ── File input label ──────────────────────────────────────────────────────
document.getElementById('csvFile').addEventListener('change', (e) => {
    const name = e.target.files[0]?.name || '';
    document.getElementById('fileName').textContent = name ? `✓  ${name}` : '';
});

// ── Form submit ───────────────────────────────────────────────────────────
document.getElementById('csvForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = document.getElementById('csvFile').files[0];
    if (!file) { alert('Please select a CSV file.'); return; }

    const formData = new FormData();
    formData.append('file', file);

    showLoading();

    try {
        const response = await fetch('/api/predict/csv', {
            method: 'POST',
            body:   formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || err.error || 'Analysis failed');
        }

        const result  = await response.json();
        currentResults = result;
        displayResults(result);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        hideLoading();
    }
});

// ── Display results ───────────────────────────────────────────────────────
function displayResults(result) {
    // Summary cards
    const { summary, total_articles } = result;
    document.getElementById('summary').innerHTML = `
        <div class="summary-box">
            <div class="summary-num">${total_articles}</div>
            <div class="summary-label">Total Articles</div>
        </div>
        <div class="summary-box highlight-fake">
            <div class="summary-num">${summary.fake_detected}</div>
            <div class="summary-label">Fake Detected</div>
        </div>
        <div class="summary-box highlight-real">
            <div class="summary-num">${summary.real_detected}</div>
            <div class="summary-label">Real Detected</div>
        </div>
        <div class="summary-box highlight-gold">
            <div class="summary-num">${summary.high_confidence}</div>
            <div class="summary-label">High Confidence</div>
        </div>
    `;

    // Table rows
    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    result.predictions.forEach(pred => {
        const isFake = pred.prediction === 1;
        const pct    = (pred.probability * 100).toFixed(1);
        const cls    = isFake ? 'fake' : 'real';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-family:var(--font-mono);font-size:0.78rem;color:var(--ink-muted);">${pred.row + 1}</td>
            <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escHtml(pred.title)}">
                ${pred.title ? escHtml(pred.title) : '<em style="color:var(--ink-faint);">—</em>'}
            </td>
            <td><span class="verdict-chip ${cls}">${pred.label}</span></td>
            <td>
                <div class="prob-bar-wrap">
                    <div class="prob-bar">
                        <div class="prob-fill ${cls}" style="width:${pct}%"></div>
                    </div>
                    <span class="prob-label">${pct}%</span>
                </div>
            </td>
            <td><span class="conf-badge ${getConfClass(pred.confidence, isFake)}">${pred.confidence}</span></td>
        `;
        tbody.appendChild(tr);
    });

    const resultsEl = document.getElementById('results');
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getConfClass(conf, isFake) {
    const map = {
        'Low':       'conf-low',
        'Medium':    'conf-medium',
        'High':      isFake ? 'conf-high'      : 'conf-real-high',
        'Very High': isFake ? 'conf-very-high' : 'conf-real-very-high',
    };
    return map[conf] || 'conf-low';
}

// ── Download sample CSV ───────────────────────────────────────────────────
function downloadSample() {
    const csv = `title,text
Breaking: Scientists discover water on Mars,NASA researchers announced today that they have found evidence of liquid water beneath the surface of Mars using advanced radar instruments aboard the Mars Reconnaissance Orbiter.
SHOCKING: Government puts microchips in vaccines to track citizens,Anonymous sources claim that secret nanobots are being injected into citizens through COVID vaccines allowing the government to monitor all activities.
New Study Links Exercise to Improved Mental Health,A comprehensive study published in the Journal of Psychiatry found that regular physical activity reduces symptoms of depression and anxiety by up to 40 percent.`;

    download(csv, 'sample_articles.csv', 'text/csv');
}

// ── Download results ──────────────────────────────────────────────────────
function downloadResults() {
    if (!currentResults) { alert('No results to download.'); return; }

    let csv = 'Row,Title,Verdict,Fake Probability (%),Confidence\n';
    currentResults.predictions.forEach(pred => {
        const pct = (pred.probability * 100).toFixed(1);
        csv += `${pred.row + 1},"${(pred.title || '').replace(/"/g, '""')}",${pred.label},${pct},${pred.confidence}\n`;
    });

    const date = new Date().toISOString().slice(0, 10);
    download(csv, `veritas_results_${date}.csv`, 'text/csv');
}

// ── Helpers ───────────────────────────────────────────────────────────────
function download(content, filename, type) {
    const blob = new Blob([content], { type });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function showLoading() {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('results').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function resetUpload() {
    document.getElementById('csvForm').reset();
    document.getElementById('fileName').textContent = '';
    document.getElementById('results').style.display = 'none';
    currentResults = null;
}
