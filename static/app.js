const TOTAL_CELLS = 9;

// Cache of the last fetched grid so saveParams can build the JSON client-side.
let gridData = [];

function fileTimestamp() {
    const d = new Date();
    const p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_` +
           `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

// Saves a blob with the system save dialog (File System Access API).
// Falls back to a regular download on Firefox/Safari.
async function saveBlobLocally(blob, suggestedName) {
    if ('showSaveFilePicker' in window) {
        try {
            const handle = await window.showSaveFilePicker({ suggestedName });
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            return;
        } catch (err) {
            if (err.name === 'AbortError') throw err; // user cancelled — abort
            // any other error: fall through to download fallback
        }
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = suggestedName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

async function fetchGrid() {
    showLoading(true);
    const res = await fetch('/api/grid');
    const data = await res.json();
    renderGrid(data.cells);
    showLoading(false);
}

function renderGrid(cells) {
    gridData = cells;
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    cells.forEach((cell, i) => {
        const cellDiv = document.createElement('div');
        cellDiv.className = 'cell';

        const img = document.createElement('img');
        img.src = cell.image || '';
        img.className = 'cell-img';
        img.onclick = () => evolve(i);

        const btnContainer = document.createElement('div');
        btnContainer.className = 'cell-buttons';

        const lockBtn = document.createElement('button');
        lockBtn.className = cell.locked ? 'btn btn-locked' : 'btn';
        lockBtn.innerText = cell.locked ? '🔒' : '🔓';
        lockBtn.onclick = () => toggleLock(i);

        const loadBtn = document.createElement('button');
        loadBtn.className = 'btn';
        loadBtn.innerText = '📂';
        loadBtn.title = 'Load Parameters';
        loadBtn.onclick = () => loadParams(i);

        const saveParamBtn = document.createElement('button');
        saveParamBtn.className = 'btn';
        saveParamBtn.innerText = '🧬';
        saveParamBtn.title = 'Save Parameters';
        saveParamBtn.onclick = () => saveParams(i);

        const saveImgBtn = document.createElement('button');
        saveImgBtn.className = 'btn';
        saveImgBtn.innerText = '🖼️';
        saveImgBtn.title = 'Save Image';
        saveImgBtn.onclick = () => saveImage(i);

        const undoBtn = document.createElement('button');
        undoBtn.className = 'btn';
        undoBtn.innerText = '↩️';
        undoBtn.title = 'Undo (restore previous genome)';
        undoBtn.onclick = () => undoCell(i);

        const uploadBtn = document.createElement('button');
        uploadBtn.className = 'btn';
        uploadBtn.innerText = '📤';
        uploadBtn.title = 'Upload & Reverse Engineer';
        uploadBtn.onclick = () => uploadImage(i);

        btnContainer.appendChild(lockBtn);
        btnContainer.appendChild(uploadBtn);
        btnContainer.appendChild(loadBtn);
        btnContainer.appendChild(saveParamBtn);
        btnContainer.appendChild(saveImgBtn);
        btnContainer.appendChild(undoBtn);

        cellDiv.appendChild(img);
        cellDiv.appendChild(btnContainer);

        grid.appendChild(cellDiv);
    });
}

async function evolve(index) {
    showLoading(true);
    await fetch('/api/evolve', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({clicked_index: index})
    });
    await fetchGrid();
}

async function generateAll() {
    showLoading(true);
    await fetch('/api/generate-all', { method: 'POST' });
    await fetchGrid();
}

async function toggleLock(index) {
    await fetch('/api/toggle-lock', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index})
    });
    await fetchGrid();
}

async function saveImage(index) {
    const size = prompt('Enter export size (WxH):', '1920x1080');
    if (!size) return;

    showLoading(true);
    try {
        const res = await fetch('/api/render-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, size })
        });
        if (!res.ok) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || `HTTP ${res.status}`);
        }
        const blob = await res.blob();
        const [w, h] = size.toLowerCase().split('x');
        const name = `image_${String(index).padStart(4, '0')}_${w}x${h}_${fileTimestamp()}.png`;
        await saveBlobLocally(blob, name);
    } catch (err) {
        if (err.name !== 'AbortError') alert('Save failed: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function undoCell(index) {
    try {
        const res = await fetch('/api/undo', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({index})
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        await fetchGrid();
    } catch (err) {
        alert('Undo failed: ' + err.message);
    }
}

async function saveParams(index) {
    // Fetch the FULL genome (luma reference included) on demand — the grid
    // response strips it to keep payloads small.
    showLoading(true);
    let genome;
    try {
        const res = await fetch('/api/get-genome', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({index})
        });
        const data = await res.json();
        showLoading(false);
        if (data.error) {
            alert(data.error);
            return;
        }
        genome = data.genome;
    } catch (err) {
        showLoading(false);
        alert('Failed to fetch genome: ' + err.message);
        return;
    }
    const saved = {
        version: '1.3',
        cell_index: index,
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
        genome: genome
    };
    const blob = new Blob([JSON.stringify(saved, null, 2)], { type: 'application/json' });
    const name = `genome_${String(index).padStart(4, '0')}_${fileTimestamp()}.json`;
    try {
        await saveBlobLocally(blob, name);
    } catch (err) {
        if (err.name !== 'AbortError') alert('Save failed: ' + err.message);
    }
}

async function loadParams(index) {
    let file = null;

    // Preferred: native open-file dialog.
    if ('showOpenFilePicker' in window) {
        try {
            const [handle] = await window.showOpenFilePicker({
                types: [{
                    description: 'Genome JSON',
                    accept: { 'application/json': ['.json'] }
                }],
                multiple: false
            });
            file = await handle.getFile();
        } catch (err) {
            if (err.name === 'AbortError') return; // user cancelled
            // other errors: fall through to fallback picker
        }
    }

    // Fallback: classic <input type="file">.
    if (!file) {
        file = await new Promise(resolve => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json,application/json';
            input.onchange = e => resolve(e.target.files[0] || null);
            input.oncancel = () => resolve(null);
            input.click();
        });
    }
    if (!file) return;

    let saved;
    try {
        saved = JSON.parse(await file.text());
    } catch (e) {
        alert('Not a valid JSON file: ' + e.message);
        return;
    }

    showLoading(true);
    try {
        const res = await fetch('/api/load-params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ index, saved })
        });
        const data = await res.json();
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            await fetchGrid();
        }
    } catch (err) {
        alert('Request failed: ' + err.message);
    } finally {
        showLoading(false);
    }
}

async function uploadImage(index) {
    // File picker MUST be triggered directly from user gesture (no prompts before it)
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*,.bmp,.png,.jpg,.jpeg,.gif';

    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // NOW ask for method (accepts number or name)
        const methodInput = prompt(
            'Select processing method:\n\n' +
            'Options:\n' +
            '  1 = brute     - Brute Force Search (tries many random genomes)\n' +
            '  2 = hillclimb - Hill Climbing Optimization (refines parameters)\n' +
            '  3 = estimate  - Frequency Analysis + Fine-tune (analyzes FFT)\n' +
            '  4 = precise   - Deterministic parameter fitting (instant, best texture match)\n' +
            '  5 = match     - Phase-preserving match (best visual similarity)\n\n' +
            'Type: 1, 2, 3 OR brute, hillclimb, estimate',
            '3'
        );
        if (!methodInput) return;

        // Normalize input (trim + lowercase)
        const cleaned = methodInput.trim().toLowerCase();

        // Map number or name to method
        let method = '';
        if (cleaned === '1' || cleaned === 'brute') {
            method = 'brute';
        } else if (cleaned === '2' || cleaned === 'hillclimb') {
            method = 'hillclimb';
        } else if (cleaned === '3' || cleaned === 'estimate') {
            method = 'estimate';
        } else if (cleaned === '4' || cleaned === 'precise') {
            method = 'precise';
        } else if (cleaned === '5' || cleaned === 'match') {
            method = 'match';
        } else {
            alert('Invalid method. Please use 1-5 or brute, hillclimb, estimate, precise, match.');
            return;
        }

        let iterations = 500;
        if (method === 'brute') {
            iterations = parseInt(prompt('Number of candidates to try (100-5000):', '1000')) || 1000;
        } else {
            iterations = parseInt(prompt('Number of optimization iterations (100-3000):', '500')) || 500;
        }

        // Upload and process
        const formData = new FormData();
        formData.append('image', file);
        formData.append('cellIndex', index.toString());
        formData.append('method', method);
        formData.append('iterations', iterations.toString());

        showLoading(true);
        const loadingText = document.getElementById('loading-text');
        if (loadingText) {
            loadingText.innerText =
                `Processing with ${method} (${iterations} iterations)...\nThis may take 10-60 seconds.`;
        }

        try {
            const res = await fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            showLoading(false);
            if (loadingText) loadingText.innerText = '';

            if (data.error) {
                alert('Error: ' + data.error);
            } else {
                await fetchGrid();
                const g = data.genome;
                alert(
                    `Reverse Engineering Complete!\n\n` +
                    `Method: ${data.method}\n` +
                    `Iterations: ${data.iterations}\n` +
                    `Similarity: ${data.similarity.toFixed(1)}%\n\n` +
                    `Genome:\n` +
                    `  Seed: ${g.seed}\n` +
                    `  Exponent: ${g.exponent.toFixed(3)}\n` +
                    `  Band Limit: ${g.band_limit.toFixed(3)}\n` +
                    `  Axis Stretch: ${g.axis_stretch.toFixed(3)}\n` +
                    `  Gamma: ${g.gamma.toFixed(3)}\n` +
                    `  Colorfulness: ${g.colorfulness.toFixed(3)}\n` +
                    `  Mutation Rate: ${g.mutation_rate.toFixed(4)}\n` +
                    `  Mutation Power: ${g.mutation_power.toFixed(2)}\n\n` +
                    `Palette:\n` +
                    `  Pal A: ${g.pal_a.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal B: ${g.pal_b.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal C: ${g.pal_c.map(v => v.toFixed(2)).join(', ')}\n` +
                    `  Pal D: ${g.pal_d.map(v => v.toFixed(2)).join(', ')}`
                );
            }
        } catch (err) {
            showLoading(false);
            if (loadingText) loadingText.innerText = '';
            alert('Request failed: ' + err.message);
        }
    };

    // Trigger file picker immediately (within user gesture)
    input.click();
}

function showLoading(show) {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div style="text-align:center;">
                <div class="spinner"></div>
                <div id="loading-text" style="margin-top:15px; color:#8b5cf6;"></div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.style.display = show ? 'flex' : 'none';
    if (!show) {
        document.getElementById('loading-text').innerText = '';
    }
}

// ============================================================================
// RENDERER FEATURE TOGGLES
// ============================================================================

const FEATURE_IDS = ['transform', 'relief', 'spikes', 'chroma', 'cone',
                     'domain_warp', 'symmetry', 'anchor_palette'];

async function loadFeatures() {
    try {
        const res = await fetch('/api/features');
        const f = await res.json();
        FEATURE_IDS.forEach(id => {
            const cb = document.getElementById('feat-' + id);
            if (cb && typeof f[id] === 'boolean') cb.checked = f[id];
        });
    } catch (err) {
        console.error('Failed to load feature settings:', err);
    }
}

function setupFeatures() {
    FEATURE_IDS.forEach(id => {
        const cb = document.getElementById('feat-' + id);
        if (!cb) return;
        cb.onchange = async () => {
            // Build the full toggle state from all checkboxes.
            const payload = {};
            FEATURE_IDS.forEach(k => {
                const el = document.getElementById('feat-' + k);
                if (el) payload[k] = el.checked;
            });
            try {
                await fetch('/api/features', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
            } catch (err) {
                alert('Failed to save feature settings: ' + err.message);
            }
        };
    });
}

document.getElementById('generate-btn').onclick = generateAll;
setupFeatures();
loadFeatures();
fetchGrid();

// ============================================================================
// ANIMATION STUDIO
// ============================================================================

async function renderAnimation() {
    const sourceA = parseInt(document.getElementById('source-a-cell').value);
    const sourceB = parseInt(document.getElementById('source-b-cell').value);
    const frames = parseInt(document.getElementById('anim-frames').value);
    const fps = parseInt(document.getElementById('anim-fps').value);
    const resolution = document.getElementById('anim-resolution').value;
    const easing = document.getElementById('anim-easing').value;
    const dir = document.getElementById('anim-dir').value;
    const mode = document.getElementById('anim-mode') ?
        document.getElementById('anim-mode').value : 'crossfade';

    // Validate resolution
    const parts = resolution.split('x');
    if (parts.length !== 2) {
        alert('Invalid resolution format. Use WxH (e.g., 1920x1080)');
        return;
    }
    const width = parseInt(parts[0]);
    const height = parseInt(parts[1]);
    if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0 || width > 8192 || height > 8192) {
        alert('Invalid resolution. Use 1-8192 pixels');
        return;
    }

    // Show progress
    const progressEl = document.getElementById('render-progress');
    const resultEl = document.getElementById('render-result');
    const statusEl = document.getElementById('render-status');
    const barEl = document.getElementById('render-bar');
    const infoEl = document.getElementById('render-info');

    progressEl.classList.remove('hidden');
    resultEl.classList.add('hidden');
    barEl.value = 0;
    statusEl.innerText = 'Preparing animation...';
    infoEl.innerText = '';

    // Send render request
    const payload = {
        source_a_cell: sourceA,
        source_b_cell: sourceB,
        frames: frames,
        fps: fps,
        width: width,
        height: height,
        easing: easing,
        dir: dir,
        mode: mode
    };

    try {
        showLoading(true);
        const res = await fetch('/api/render-animation', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        showLoading(false);
        progressEl.classList.add('hidden');

        if (res.ok && data.status === 'rendered') {
            resultEl.className = 'result-message success';
            resultEl.innerHTML = `
                <strong>✅ Animation Rendered Successfully!</strong>
                Output: ${data.output_dir}
                Total Frames: ${data.total_frames}
                Duration: ${(data.total_frames/fps).toFixed(1)}s @ ${fps}fps
                JSON Metadata: ${data.json_path}

                Next Steps:
                • Import frames into video editor (Premiere, Davinci, etc.)
                • Or use FFmpeg to encode: ffmpeg -framerate ${fps} -i "${data.output_dir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p output.mp4
            `;
            console.log('Animation rendered:', data);
        } else {
            throw new Error(data.error || 'Rendering failed');
        }
    } catch (err) {
        showLoading(false);
        progressEl.classList.add('hidden');
        resultEl.className = 'result-message error';
        resultEl.innerHTML = `<strong>❌ Error:</strong>\n${err.message}`;
        console.error('Animation error:', err);
    }
}

document.getElementById('render-btn').onclick = renderAnimation;
