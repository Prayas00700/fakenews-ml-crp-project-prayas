// ── Character counter ─────────────────────────────────────────────────────
const textArea  = document.getElementById('text');
const charCount = document.getElementById('charCount');

if (textArea && charCount) {
    textArea.addEventListener('input', () => {
        charCount.textContent = textArea.value.length.toLocaleString();
    });
}

// ── Form submit ───────────────────────────────────────────────────────────
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const title = document.getElementById('title').value.trim();
    const text  = document.getElementById('text').value.trim();

    if (!text) { alert('Please paste the article body.'); return; }

    showLoading();

    try {
        const response = await fetch('/api/predict', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ title, text }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || err.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResult(result);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        hideLoading();
    }
});

// ── Display result ────────────────────────────────────────────────────────
function displayResult(result) {
    const isFake     = result.prediction === 1;
    const pct        = (result.probability * 100).toFixed(1);
    const confClass  = getConfClass(result.confidence, isFake);
    const rec        = getRecommendation(isFake, result.confidence);

    document.getElementById('resultContent').innerHTML = `
        <div class="verdict-block ${isFake ? 'fake' : 'real'}">
            <div class="verdict-icon">${isFake ? '🚨' : '✅'}</div>
            <div>
                <div class="verdict-title">${isFake ? 'Likely Fake' : 'Likely Real'}</div>
                <div class="verdict-sub">${result.interpretation}</div>
            </div>
        </div>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-value">${isFake ? pct + '%' : (100 - parseFloat(pct)).toFixed(1) + '%'}</div>
                <div class="metric-label">${isFake ? 'Fake' : 'Real'} probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${result.label}</div>
                <div class="metric-label">Verdict</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="font-size:1.2rem;">
                    <span class="conf-badge ${confClass}">${result.confidence}</span>
                </div>
                <div class="metric-label">Confidence</div>
            </div>
        </div>

        <div class="recommendation-box">
            <h4>📌 Recommendation</h4>
            <p>${rec}</p>
        </div>
    `;

    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

function getRecommendation(isFake, confidence) {
    if (!isFake && (confidence === 'Low' || confidence === 'Medium')) {
        return 'The model leans toward this being real, but confidence is modest. Cross-check with established news outlets before sharing.';
    }
    if (!isFake) {
        return 'This article shows strong signals of being authentic. Still, verify key claims through official sources or reputable fact-checkers.';
    }
    if (confidence === 'Low' || confidence === 'Medium') {
        return 'Possible misinformation detected, though confidence is limited. Investigate the source and check whether the claims appear elsewhere.';
    }
    return 'Strong indicators of fabricated or misleading content. Do not share without independent verification. Check Snopes, FactCheck.org, or AP Fact Check.';
}

// ── Helpers ───────────────────────────────────────────────────────────────
function showLoading() {
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('results').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').style.display = 'none';
    if (charCount) charCount.textContent = '0';
}
