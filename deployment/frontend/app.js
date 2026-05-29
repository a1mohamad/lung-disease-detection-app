const API_BASE = window.LUNG_API_BASE || "";

const THRESHOLDS = {
  densenet: 0.91,
  efficientnet: 0.85,
  inception_v3: 0.5,
  mobilenet_v3: 0.5,
};

const modeButtons = document.querySelectorAll(".segment");
const modeViews = document.querySelectorAll(".mode-view");
const form = document.querySelector("#predictForm");
const fileInput = document.querySelector("#imageFile");
const fileMeta = document.querySelector("#fileMeta");
const analyzeButton = document.querySelector("#analyzeButton");
const resetButton = document.querySelector("#resetButton");
const messagePanel = document.querySelector("#messagePanel");
const apiStatus = document.querySelector("#apiStatus");

// Panels & Modals
const pPrimary = document.querySelector("#resultPrimary");
const pPipeline = document.querySelector("#resultPipeline");
const pEnsemble = document.querySelector("#resultEnsemble");
const lightboxOverlay = document.querySelector("#lightboxOverlay");
const lightboxImage = document.querySelector("#lightboxImage");
const lightboxClose = document.querySelector("#lightboxClose");

let currentMode = "upload";

function setMode(mode) {
  currentMode = mode;
  modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  modeViews.forEach((view) => view.classList.toggle("active", view.dataset.view === mode));
  clearMessage();
}

function asPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "0.0%";
  return `${(n * 100).toFixed(1)}%`;
}

function setMessage(text) {
  messagePanel.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${text}`;
  messagePanel.classList.remove("hidden");
}

function clearMessage() {
  messagePanel.textContent = "";
  messagePanel.classList.add("hidden");
}

function absoluteUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path}`;
}

function setImage(id, path) {
  const img = document.querySelector(id);
  if (img) img.src = absoluteUrl(path);
}

// Lightbox Logic
document.querySelectorAll('.zoomable').forEach(img => {
    img.addEventListener('click', (e) => {
        if(!e.target.src) return;
        lightboxImage.src = e.target.src;
        lightboxOverlay.classList.remove('hidden');
    });
});
lightboxClose.addEventListener('click', () => lightboxOverlay.classList.add('hidden'));
lightboxOverlay.addEventListener('click', (e) => {
    if(e.target === lightboxOverlay) lightboxOverlay.classList.add('hidden');
});

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error();
    apiStatus.innerHTML = `<div class="pulse-dot"></div> Telemetry Online`;
    apiStatus.classList.add("ok");
    apiStatus.classList.remove("bad");
  } catch {
    apiStatus.innerHTML = `<div class="pulse-dot"></div> Telemetry Offline`;
    apiStatus.classList.add("bad");
    apiStatus.classList.remove("ok");
  }
}

function buildRequest() {
  if (currentMode === "upload") {
    const file = fileInput.files[0];
    if (!file) throw new Error("Please select an X-Ray file.");
    const data = new FormData();
    data.append("file", file);
    return { url: `${API_BASE}/predict/upload?return_all=true`, options: { method: "POST", body: data } };
  }
  if (currentMode === "url") {
    const imageUrl = document.querySelector("#imageUrl").value.trim();
    if (!imageUrl) throw new Error("Please paste a valid image URL.");
    return { url: `${API_BASE}/predict?return_all=true`, options: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_url: imageUrl }) } };
  }
  const imageBase64 = document.querySelector("#imageBase64").value.trim();
  if (!imageBase64) throw new Error("Please paste a Base64 string.");
  return { url: `${API_BASE}/predict?return_all=true`, options: { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_base64: imageBase64 }) } };
}

async function submitPrediction(event) {
  event.preventDefault();
  clearMessage();
  analyzeButton.disabled = true;
  analyzeButton.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Matrix...`;

  try {
    const request = buildRequest();
    const response = await fetch(request.url, request.options);
    const payload = await response.json();

    if (!response.ok) throw new Error(payload.message || payload.detail || "Analysis failed.");
    renderResult(payload);
  } catch (error) {
    setMessage(error.message || "Analysis failed.");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = "Execute Analysis";
  }
}

function renderResult(result) {
  // 1. Primary Verdict
  const label = result.final_label_name || `Class ${result.final_label}`;
  const isUnhealthy = String(label).toLowerCase() === "unhealthy" || result.final_label === 1;
  const badge = document.querySelector("#decisionBadge");

  document.querySelector("#finalLabel").textContent = label.toUpperCase();
  badge.textContent = isUnhealthy ? "ACTION REQUIRED" : "CLEAR";
  badge.classList.toggle("unhealthy", isUnhealthy);
  badge.classList.toggle("healthy", !isUnhealthy);

  // 2. Dual Probabilities
  const healthyProb = result.final_probs_by_label?.healthy || (isUnhealthy ? 1 - result.final_prob : result.final_prob) || 0;
  const unhealthyProb = result.final_probs_by_label?.unhealthy || (isUnhealthy ? result.final_prob : 1 - result.final_prob) || 0;
  
  document.querySelector("#healthyScore").textContent = asPercent(healthyProb);
  document.querySelector("#unhealthyScore").textContent = asPercent(unhealthyProb);
  
  // Update Clinical Notes
  const notesText = document.querySelector("#clinicalNotesText");
  const notesBox = document.querySelector("#clinicalNotesBox");
  if (isUnhealthy) {
      notesText.textContent = "Pathological markers detected in the structural matrix. Recommend immediate radiological review and clinical correlation. Patient may require further diagnostic testing.";
      notesBox.style.borderLeftColor = "var(--pathology-accent)";
  } else {
      notesText.textContent = "No significant pathological markers detected. Matrix indicates clear respiratory structures. Routine preventive care recommended based on patient history.";
      notesBox.style.borderLeftColor = "var(--healthy-accent)";
  }

  // Animate Gauge (Speedometer)
  const gaugeFill = document.querySelector("#gaugeFill");
  const gaugeReadout = document.querySelector("#gaugeReadout");
  const gaugeOffset = 126 - (126 * unhealthyProb);
  
  setTimeout(() => {
      document.querySelector("#healthyBar").style.width = `${Math.min(100, healthyProb * 100)}%`;
      document.querySelector("#unhealthyBar").style.width = `${Math.min(100, unhealthyProb * 100)}%`;
      
      gaugeFill.style.strokeDashoffset = gaugeOffset;
      gaugeFill.style.stroke = isUnhealthy ? "var(--pathology-accent)" : "var(--healthy-accent)";
      gaugeReadout.textContent = asPercent(unhealthyProb);
      gaugeReadout.style.color = isUnhealthy ? "var(--pathology-accent)" : "var(--healthy-accent)";
  }, 50);

  // 3. Pipeline Images
  setImage("#sourceImage", result.source_url);
  setImage("#roiImage", result.roi_url);
  setImage("#maskImage", result.mask_url);
  setImage("#overlayImage", result.overlay_url);

  // 4. Disease & Ensemble
  renderDisease(result.disease);
  renderModels(result.models_results || {});

  pPrimary.classList.remove("hidden");
  pPipeline.classList.remove("hidden");
  pEnsemble.classList.remove("hidden");
}

function renderDisease(disease) {
  const card = document.querySelector("#diseaseCard");
  const scores = document.querySelector("#diseaseScores");
  scores.innerHTML = "";
  if (!disease) { card.classList.add("hidden"); return; }
  document.querySelector("#diseaseLabel").textContent = disease.label_name || `Sub-class ${disease.label}`;
  Object.entries(disease.probs_by_label || {}).forEach(([name, value]) => {
    const row = document.createElement("div");
    row.className = "mini-score-row";
    row.innerHTML = `<span>${name.toUpperCase()}</span> <strong>${asPercent(value)}</strong>`;
    scores.appendChild(row);
  });
  card.classList.remove("hidden");
}

// Fixed function: Removes conditional flipping logic so probabilities map directly to 1 (Pathology)
function renderModels(models) {
  const container = document.querySelector("#modelDetails");
  container.innerHTML = "";

  Object.entries(models).forEach(([name, model]) => {
    const threshold = THRESHOLDS[name] || 0.5;
    
    // model.prob from the API is strictly the Pathology (class 1) raw probability
    let mUnhealthyProb = model.prob;
    let mHealthyProb = 1 - model.prob;

    const row = document.createElement("div");
    row.className = "model-row";
    
    row.innerHTML = `
      <div class="model-header-row">
        <div>
          <span class="model-name">${formatModelName(name)}</span>
          <span class="model-meta">Vote: ${model.label_name || model.label} &bull; Thr: ${threshold}</span>
        </div>
      </div>
      <div class="model-dual-bars">
        <div>
            <div class="mini-prob-label h-text"><span>Healthy</span><span>${asPercent(mHealthyProb)}</span></div>
            <div class="mini-prob-bar-bg"><div class="mini-prob-fill healthy-fill" style="width: ${mHealthyProb * 100}%"></div></div>
        </div>
        <div>
            <div class="mini-prob-label u-text"><span>Pathology</span><span>${asPercent(mUnhealthyProb)}</span></div>
            <div class="mini-prob-bar-bg"><div class="mini-prob-fill pathology-fill" style="width: ${mUnhealthyProb * 100}%"></div></div>
        </div>
      </div>
    `;
    container.appendChild(row);
  });
}

function formatModelName(name) {
  return name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
}

function resetUi() {
  form.reset();
  fileMeta.textContent = "PNG, JPG, WEBP";
  
  // Clear the image preview
  const preview = document.querySelector("#uploadPreview");
  preview.src = "";
  preview.classList.add("hidden");
  
  pPrimary.classList.add("hidden");
  pPipeline.classList.add("hidden");
  pEnsemble.classList.add("hidden");
  document.querySelector("#healthyBar").style.width = "0%";
  document.querySelector("#unhealthyBar").style.width = "0%";
  document.querySelector("#gaugeFill").style.strokeDashoffset = "126";
  document.querySelector("#gaugeReadout").textContent = "0%";
  clearMessage();
}

modeButtons.forEach((btn) => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

// Added logic to inject an image preview via Object URL
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  const preview = document.querySelector("#uploadPreview");
  if (file) {
    fileMeta.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  } else {
    fileMeta.textContent = "PNG, JPG, WEBP";
    preview.src = "";
    preview.classList.add("hidden");
  }
});

form.addEventListener("submit", submitPrediction);
resetButton.addEventListener("click", resetUi);

checkApi();