// ── Character counter ─────────────────────────────────────────────────────
const textArea  = document.getElementById('text');
const charCount = document.getElementById('charCount');

if (textArea && charCount) {
    textArea.addEventListener('input', () => {
        charCount.textContent = textArea.value.length.toLocaleString();
    });
}

// ── Form submit — fast predict ────────────────────────────────────────────
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const title = document.getElementById('title').value.trim();
    const text  = document.getElementById('text').value.trim();
    if (!text) { alert('Please paste the article body.'); return; }

    showLoading('Scanning article\u2026');
    try {
        const response = await fetch('/api/predict', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ title, text }),
        });
        const result = await response.json();
        if (!response.ok || result.detail || result.error) {
            throw new Error(result.detail || result.error || 'Prediction failed');
        }
        displayResult(result, false);
    } catch (err) {
        alert('Backend Error:\n' + err.message);
    } finally {
        hideLoading();
    }
});

// ── Explain button — SHAP ─────────────────────────────────────────────────
async function explainArticle() {
    const title = document.getElementById('title').value.trim();
    const text  = document.getElementById('text').value.trim();
    if (!text) { alert('Please paste the article body first.'); return; }

    showLoading('Running SHAP explanation\u2026 this takes 30\u201360 seconds.');
    try {
        const response = await fetch('/api/explain', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ title, text }),
        });
        const result = await response.json();
        if (!response.ok || result.detail || result.error) {
            throw new Error(result.detail || result.error || 'Explanation failed');
        }
        displayResult(result, true);
    } catch (err) {
        alert('Explanation Error:\n' + err.message);
    } finally {
        hideLoading();
    }
}

// ── Display result ────────────────────────────────────────────────────────
function displayResult(result, withExplanation) {
    const predictionVal  = result.prediction  ?? result.Prediction;
    const probabilityVal = result.probability ?? result.Probability ?? result.score;
    const labelVal       = result.label       ?? result.Label;

    if (predictionVal === undefined && probabilityVal === undefined) {
        document.getElementById('resultContent').innerHTML = `
            <div class="verdict-block fake">
                <div class="verdict-icon">\u26a0\ufe0f</div>
                <div>
                    <div class="verdict-title">Data Mismatch</div>
                    <div class="verdict-sub">Unexpected JSON structure from backend.</div>
                </div>
            </div>
            <pre style="background:var(--surface-2);padding:16px;border-radius:var(--radius);font-size:0.85rem;color:var(--accent);overflow-x:auto;">${JSON.stringify(result, null, 2)}</pre>
        `;
        showResults();
        return;
    }

    const isFake       = (predictionVal == 1 || String(labelVal).toLowerCase() === 'fake');
    const isUnverified = String(labelVal).toUpperCase() === 'UNVERIFIED';
    const fakeProbVal  = parseFloat(result.fake_prob ?? probabilityVal ?? 0);
    const realProbVal  = parseFloat(result.real_prob ?? (1 - fakeProbVal));
    const pct          = isNaN(fakeProbVal) ? 'N/A' : (fakeProbVal * 100).toFixed(1);
    const realPct      = isNaN(realProbVal) ? 'N/A' : (realProbVal * 100).toFixed(1);
    const interpretationVal = result.interpretation ?? 'No interpretation provided.';
    const confidenceVal     = result.confidence     ?? 'UNDEFINED';
    const confClass         = getConfClass(confidenceVal, isFake);
    const rec               = getRecommendation(isFake, confidenceVal);

    const verdictClass = isFake ? 'fake' : (isUnverified ? 'unverified' : 'real');
    const verdictIcon  = isFake ? '\ud83d\udea8' : (isUnverified ? '\u26a0\ufe0f' : '\u2705');
    const verdictTitle = isFake ? 'Likely Fake' : (isUnverified ? 'Unverified' : 'Likely Real');

    let explanationHtml = '';
    if (withExplanation && result.explanation && result.explanation.length > 0) {
        explanationHtml = buildExplanationHtml(result.explanation);
    } else if (!withExplanation) {
        explanationHtml = `
            <div class="explain-prompt">
                <p>Want to know <strong>why</strong> the model made this decision?</p>
                <button class="btn-explain" onclick="explainArticle()">
                    <span class="btn-explain-main">\ud83d\udd0d Explain This Verdict</span>
                    <span class="btn-explain-sub">SHAP \u00b7 ~30\u201360 seconds</span>
                </button>
            </div>
        `;
    }

    document.getElementById('resultContent').innerHTML = `
        <div class="verdict-block ${verdictClass}">
            <div class="verdict-icon">${verdictIcon}</div>
            <div>
                <div class="verdict-title">${verdictTitle}</div>
                <div class="verdict-sub">${interpretationVal}</div>
            </div>
        </div>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-value">${isFake ? pct : realPct}%</div>
                <div class="metric-label">${isFake ? 'Fake' : 'Real'} probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${labelVal || 'UNDEFINED'}</div>
                <div class="metric-label">Verdict</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="font-size:1.2rem;">
                    <span class="conf-badge ${confClass}">${confidenceVal}</span>
                </div>
                <div class="metric-label">Confidence</div>
            </div>
        </div>

        <div class="recommendation-box">
            <h4>\ud83d\udccc Recommendation</h4>
            <p>${rec}</p>
        </div>

        ${explanationHtml}
    `;

    showResults();
}

// ── Build SHAP explanation HTML ───────────────────────────────────────────
function buildExplanationHtml(explanation) {
    const fakeWords = explanation.filter(e => e.direction === 'fake').slice(0, 6);
    const realWords = explanation.filter(e => e.direction === 'real').slice(0, 6);
    const maxScore  = Math.max(...explanation.map(e => Math.abs(e.score)), 0.001);

    function wordChip(entry) {
        const barWidth = Math.max(Math.round((Math.abs(entry.score) / maxScore) * 100), 10);
        const sign     = entry.score > 0 ? '+' : '';
        const cls      = entry.direction === 'fake' ? 'chip-fake' : 'chip-real';
        return `
            <div class="word-chip ${cls}">
                <span class="chip-word">${escHtml(entry.word)}</span>
                <div class="chip-bar-track">
                    <div class="chip-bar-fill" style="width:${barWidth}%"></div>
                </div>
                <span class="chip-score">${sign}${(entry.score * 100).toFixed(1)}</span>
            </div>
        `;
    }

    const fakeCol = fakeWords.length > 0 ? `
        <div class="explain-col">
            <div class="explain-col-head fake-head">
                <span class="head-dot fake-dot"></span>
                Pushed toward FAKE
                <div class="head-sub">Positive SHAP value \u2014 increases fake probability</div>
            </div>
            <div class="chip-list">${fakeWords.map(wordChip).join('')}</div>
        </div>
    ` : '';

    const realCol = realWords.length > 0 ? `
        <div class="explain-col">
            <div class="explain-col-head real-head">
                <span class="head-dot real-dot"></span>
                Pushed toward REAL
                <div class="head-sub">Negative SHAP value \u2014 decreases fake probability</div>
            </div>
            <div class="chip-list">${realWords.map(wordChip).join('')}</div>
        </div>
    ` : '';

    return `
        <div class="explanation-box">
            <div class="explanation-header">
                <span class="explanation-title">\ud83d\udd0d Why did the model decide this?</span>
                <span class="explanation-badge">KernelSHAP</span>
            </div>
            <p class="explanation-desc">
                Each word&rsquo;s SHAP value shows its average contribution to the fake probability,
                computed by measuring how predictions change when the word is included or excluded
                across many random combinations of words (Shapley coalitions).
            </p>
            <div class="explain-grid">${fakeCol}${realCol}</div>
        </div>
    `;
}

// ── Helpers ───────────────────────────────────────────────────────────────
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
    if (!isFake && (confidence === 'Low' || confidence === 'Medium'))
        return 'The model leans toward this being real, but confidence is modest. Cross-check with established news outlets before sharing.';
    if (!isFake)
        return 'This article shows strong signals of being authentic. Still, verify key claims through official sources or reputable fact-checkers.';
    if (confidence === 'Low' || confidence === 'Medium')
        return 'Possible misinformation detected, though confidence is limited. Investigate the source and check whether the claims appear elsewhere.';
    return 'Strong indicators of fabricated or misleading content. Do not share without independent verification. Check Snopes, FactCheck.org, or AP Fact Check.';
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function showLoading(msg) {
    const el = document.querySelector('.loading-text');
    if (el) el.textContent = msg || 'Scanning article\u2026';
    document.getElementById('loading').style.display  = 'flex';
    document.getElementById('results').style.display  = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showResults() {
    const d = document.getElementById('results');
    d.style.display = 'block';
    d.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').style.display = 'none';
    if (charCount) charCount.textContent = '0';
}
