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

        const result = await response.json();

        // Failsafe: Catch HTTP errors OR backend errors masquerading as 200 OK
        if (!response.ok || result.detail || result.error) {
            throw new Error(result.detail || result.error || 'Prediction failed');
        }

        displayResult(result);
    } catch (err) {
        alert('Backend Error:\n' + err.message);
    } finally {
        hideLoading();
    }
});

// ── Display result ────────────────────────────────────────────────────────
function displayResult(result) {
    console.log("Raw API Response:", result); // Helpful for browser console debugging

    // Extract values with fallbacks for capitalized/alternative keys
    const predictionVal = result.prediction ?? result.Prediction;
    const probabilityVal = result.probability ?? result.Probability ?? result.score;
    const labelVal = result.label ?? result.Label;
    
    // FAILSAFE: If the essential data is entirely missing or nested differently
    if (predictionVal === undefined && probabilityVal === undefined) {
        document.getElementById('resultContent').innerHTML = `
            <div class="verdict-block fake">
                <div class="verdict-icon">⚠️</div>
                <div>
                    <div class="verdict-title">Data Mismatch</div>
                    <div class="verdict-sub">The Python backend returned an unexpected JSON structure. Ensure your Python API returns exact lowercase keys: 'prediction', 'probability', 'label', 'confidence', 'interpretation'.</div>
                </div>
            </div>
            <div style="background:var(--surface-2); padding:16px; border-radius:var(--radius); margin-top:16px; font-family:var(--font-mono); font-size:0.85rem; overflow-x:auto; color:var(--accent);">
                <strong>Raw Backend Output:</strong><br><br>
                <pre>${JSON.stringify(result, null, 2)}</pre>
            </div>
        `;
        const resultsDiv = document.getElementById('results');
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
    }

    // Safely parse the values
    const isFake = (predictionVal == 1 || String(labelVal).toLowerCase() === 'fake');
    const fakeProbVal = parseFloat(result.fake_prob ?? probabilityVal ?? 0);
    const realProbVal = parseFloat(result.real_prob ?? (1 - fakeProbVal));
    const pct = isNaN(fakeProbVal) ? 'NaN' : (fakeProbVal * 100).toFixed(1);
    const realPct = isNaN(realProbVal) ? 'NaN' : (realProbVal * 100).toFixed(1);
    
    const interpretationVal = result.interpretation ?? result.Interpretation ?? 'No interpretation provided.';
    const confidenceVal = result.confidence ?? result.Confidence ?? 'UNDEFINED';
    const confClass  = getConfClass(confidenceVal, isFake);
    const rec        = getRecommendation(isFake, confidenceVal);

    // Render the normal UI
    document.getElementById('resultContent').innerHTML = `
        <div class="verdict-block ${isFake ? 'fake' : 'real'}">
            <div class="verdict-icon">${isFake ? '🚨' : '✅'}</div>
            <div>
                <div class="verdict-title">${isFake ? 'Likely Fake' : 'Likely Real'}</div>
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