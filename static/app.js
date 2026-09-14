/**
 * app.js - Client logic for BodyMaps CT Segmentation Viewer
 */

// Application State
const state = {
  currentCase: "BDMAP_00000338",
  orientation: "axial",
  windowPreset: "abdomen",
  sliceIdx: 35,
  maxSlices: 71,
  opacity: 0.55,
  activeOrgans: [1, 2, 3, 4, 6, 7, 8, 9, 11],
  metadata: null,
  allSelected: true,
  health: null,
};

// DOM Elements
const sliceImg = document.getElementById("sliceImage");
const sliceSlider = document.getElementById("sliceSlider");
const sliceVal = document.getElementById("sliceVal");
const opacitySlider = document.getElementById("opacitySlider");
const opacityVal = document.getElementById("opacityVal");
const caseSelect = document.getElementById("caseSelect");
const uploadDropZone = document.getElementById("uploadDropZone");
const fileInput = document.getElementById("fileInput");
const btnRunInfer = document.getElementById("btnRunInfer");
const organList = document.getElementById("organList");
const btnToggleAll = document.getElementById("btnToggleAll");
const spinner = document.getElementById("spinner");
const spinnerText = document.getElementById("spinnerText");
const hardwareBadge = document.getElementById("hardwareBadge");

const displayCaseId = document.getElementById("displayCaseId");
const displaySlice = document.getElementById("displaySlice");
const displayDim = document.getElementById("displayDim");
const displayWindow = document.getElementById("displayWindow");
const displayDice = document.getElementById("displayDice");
const displaySpacing = document.getElementById("displaySpacing");
const inferStatusText = document.getElementById("inferStatusText");
const inferProvenanceText = document.getElementById("inferProvenanceText");
const btnDownloadNii = document.getElementById("btnDownloadNii");
const btnDownloadZip = document.getElementById("btnDownloadZip");

const statusRibbon = document.getElementById("statusRibbon");
const statusRibbonDot = document.getElementById("statusRibbonDot");
const statusRibbonText = document.getElementById("statusRibbonText");

const errorBanner = document.getElementById("errorBanner");
const errorBannerText = document.getElementById("errorBannerText");
const btnDismissError = document.getElementById("btnDismissError");

const accuracyPanel = document.getElementById("accuracyPanel");
const accuracyTableBody = document.getElementById("accuracyTableBody");

const btnHowItWorks = document.getElementById("btnHowItWorks");
const btnCloseHowItWorks = document.getElementById("btnCloseHowItWorks");
const howItWorksBackdrop = document.getElementById("howItWorksBackdrop");

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  checkHardware();
  loadCases();
  loadCaseMeta(state.currentCase);
  setupEvents();
});

function showError(message) {
  errorBannerText.textContent = message;
  errorBanner.style.display = "flex";
}

function hideError() {
  errorBanner.style.display = "none";
}

function setStatusRibbon(kind, text) {
  // kind: "prediction" | "ground-truth" | "none" | "warning"
  statusRibbon.className = "status-ribbon status-ribbon-" + kind;
  statusRibbonText.textContent = text;
}

function checkHardware() {
  fetch("/api/health")
    .then((r) => r.json())
    .then((data) => {
      state.health = data;
      if (!data.checkpoint_exists) {
        hardwareBadge.className = "badge badge-danger";
        hardwareBadge.textContent = "Checkpoint missing";
        showError(
          "Model checkpoint file is missing on the server (supervised_suprem_unet_2100.pth). " +
          "Inference cannot run until it is downloaded. See README for the download link."
        );
        return;
      }
      if (data.accelerated) {
        hardwareBadge.className = "badge badge-success";
        hardwareBadge.textContent = `GPU: ${data.device_label}`;
      } else {
        hardwareBadge.className = "badge badge-warning";
        hardwareBadge.textContent = `CPU only — inference will be slow`;
      }
    })
    .catch(() => {
      hardwareBadge.className = "badge badge-danger";
      hardwareBadge.textContent = "Backend offline";
      showError("Could not reach the backend at /api/health. Is the server running?");
    });
}

function loadCases() {
  fetch("/api/cases")
    .then((r) => r.json())
    .then((data) => {
      caseSelect.innerHTML = "";
      data.cases.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c.case_id;
        opt.textContent = c.title;
        caseSelect.appendChild(opt);
      });
      caseSelect.value = state.currentCase;
    })
    .catch(() => showError("Could not load the list of available cases."));
}

function loadCaseMeta(caseId) {
  showSpinner("Loading volume metadata...");
  fetch(`/api/case/${caseId}/meta`)
    .then((r) => {
      if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || "Failed to load case"); });
      return r.json();
    })
    .then((meta) => {
      state.metadata = meta;
      state.maxSlices = meta.slices_axial;
      state.sliceIdx = Math.floor(meta.slices_axial / 2);

      updateSliderLimits();
      renderOrganList(meta.organs || []);
      renderAccuracyPanel(meta);
      renderProvenance(meta);
      renderStatusRibbon(caseId, meta);

      displayCaseId.textContent = caseId;
      displayDim.textContent = `${meta.shape[0]} × ${meta.shape[1]} px`;
      displaySpacing.textContent = `Spacing: ${meta.spacing_mm.join(" × ")} mm`;

      if (meta.mean_dice != null) {
        displayDice.textContent = `Mean Dice: ${meta.mean_dice}`;
        displayDice.style.display = "block";
      } else {
        displayDice.style.display = "none";
      }

      btnDownloadNii.href = `/api/case/${caseId}/download/nii`;
      btnDownloadZip.href = `/api/case/${caseId}/download/zip`;

      updateSliceView();
      hideSpinner();
      hideError();
    })
    .catch((err) => {
      hideSpinner();
      showError("Error loading case metadata: " + err.message);
    });
}

function renderStatusRibbon(caseId, meta) {
  if (!meta.has_segmentation) {
    setStatusRibbon("none", `No segmentation yet for ${caseId}. Click "Run SuPreM Segmentation" to compute one.`);
    return;
  }
  const info = meta.inference_info;
  if (info) {
    const when = new Date(info.computed_at_utc).toLocaleString();
    setStatusRibbon(
      "prediction",
      `Showing a LIVE model prediction — computed by SuPreM UNet on ${info.device_label} in ${info.elapsed_seconds}s (${when}).`
    );
  } else {
    // Segmentation exists but has no inference_meta.json sidecar: this only
    // happens for the bundled demo case before its first real inference run
    // in this environment, or for hand-provided data.
    setStatusRibbon(
      "warning",
      `Showing a bundled result with no recorded provenance for this case. Click "Run SuPreM Segmentation" to generate a fresh, timestamped prediction.`
    );
  }
}

function renderProvenance(meta) {
  const info = meta.inference_info;
  if (!info) {
    inferProvenanceText.textContent = "";
    return;
  }
  const when = new Date(info.computed_at_utc).toLocaleString();
  inferProvenanceText.textContent = `Last run: ${info.device_label} · ${info.elapsed_seconds}s · ${when}`;
}

function renderAccuracyPanel(meta) {
  if (!meta.has_groundtruth_comparison) {
    accuracyPanel.style.display = "none";
    return;
  }
  accuracyPanel.style.display = "block";
  accuracyTableBody.innerHTML = "";
  (meta.organs || []).forEach((organ) => {
    if (!organ.present || organ.dice == null) return;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${organ.label}</td>
      <td>${organ.volume_ml != null ? organ.volume_ml.toFixed(1) : "—"}</td>
      <td>${organ.gt_volume_ml != null ? organ.gt_volume_ml.toFixed(1) : "—"}</td>
      <td class="dice-cell">${organ.dice.toFixed(3)}</td>
    `;
    accuracyTableBody.appendChild(tr);
  });
}

function updateSliderLimits() {
  let max = state.metadata ? state.metadata.slices_axial : 71;
  if (state.orientation === "coronal") max = state.metadata.slices_coronal;
  if (state.orientation === "sagittal") max = state.metadata.slices_sagittal;

  sliceSlider.max = max - 1;
  state.sliceIdx = Math.min(state.sliceIdx, max - 1);
  sliceSlider.value = state.sliceIdx;
  sliceVal.textContent = state.sliceIdx;
}

function updateSliceView() {
  const orientLabel = state.orientation.charAt(0).toUpperCase() + state.orientation.slice(1);
  const total = parseInt(sliceSlider.max) + 1;
  displaySlice.textContent = `${orientLabel} Slice: ${state.sliceIdx} / ${total}`;
  sliceVal.textContent = state.sliceIdx;

  const organsParam = state.activeOrgans.join(",");
  const url = `/api/case/${state.currentCase}/slice/${state.sliceIdx}?orientation=${state.orientation}&window=${state.windowPreset}&opacity=${state.opacity}&organs=${organsParam}`;

  sliceImg.src = url;
}

function renderOrganList(organs) {
  organList.innerHTML = "";
  organs.forEach((organ) => {
    const item = document.createElement("div");
    item.className = "organ-item";

    const left = document.createElement("div");
    left.className = "organ-left";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.activeOrgans.includes(organ.id);
    cb.addEventListener("change", (e) => {
      e.stopPropagation();
      toggleOrgan(organ.id, cb.checked);
    });

    const chip = document.createElement("span");
    chip.className = "organ-chip";
    chip.style.backgroundColor = organ.color;

    const nameSpan = document.createElement("span");
    nameSpan.textContent = organ.label;

    left.appendChild(cb);
    left.appendChild(chip);
    left.appendChild(nameSpan);

    const right = document.createElement("div");
    right.className = "organ-meta";

    if (organ.volume_ml) {
      right.textContent = `${organ.volume_ml} mL`;
    }
    if (organ.dice != null) {
      const diceSpan = document.createElement("span");
      diceSpan.className = "dice-pill";
      diceSpan.textContent = `Dice: ${organ.dice}`;
      right.appendChild(diceSpan);
    }

    item.appendChild(left);
    item.appendChild(right);

    item.addEventListener("click", () => {
      cb.checked = !cb.checked;
      toggleOrgan(organ.id, cb.checked);
    });

    organList.appendChild(item);
  });
}

function toggleOrgan(organId, isChecked) {
  if (isChecked && !state.activeOrgans.includes(organId)) {
    state.activeOrgans.push(organId);
  } else if (!isChecked) {
    state.activeOrgans = state.activeOrgans.filter((id) => id !== organId);
  }
  updateSliceView();
}

function setupEvents() {
  // Slice Slider
  sliceSlider.addEventListener("input", (e) => {
    state.sliceIdx = parseInt(e.target.value);
    updateSliceView();
  });

  // Opacity Slider
  opacitySlider.addEventListener("input", (e) => {
    state.opacity = parseInt(e.target.value) / 100.0;
    opacityVal.textContent = `${e.target.value}%`;
    updateSliceView();
  });

  // Step buttons
  document.getElementById("btnPrevSlice").addEventListener("click", () => {
    if (state.sliceIdx > 0) {
      state.sliceIdx--;
      sliceSlider.value = state.sliceIdx;
      updateSliceView();
    }
  });

  document.getElementById("btnNextSlice").addEventListener("click", () => {
    if (state.sliceIdx < parseInt(sliceSlider.max)) {
      state.sliceIdx++;
      sliceSlider.value = state.sliceIdx;
      updateSliceView();
    }
  });

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      if (state.sliceIdx > 0) {
        state.sliceIdx--;
        sliceSlider.value = state.sliceIdx;
        updateSliceView();
      }
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      if (state.sliceIdx < parseInt(sliceSlider.max)) {
        state.sliceIdx++;
        sliceSlider.value = state.sliceIdx;
        updateSliceView();
      }
    }
  });

  // Mouse wheel navigation on viewport
  document.getElementById("viewport").addEventListener("wheel", (e) => {
    e.preventDefault();
    if (e.deltaY > 0 && state.sliceIdx < parseInt(sliceSlider.max)) {
      state.sliceIdx++;
      sliceSlider.value = state.sliceIdx;
      updateSliceView();
    } else if (e.deltaY < 0 && state.sliceIdx > 0) {
      state.sliceIdx--;
      sliceSlider.value = state.sliceIdx;
      updateSliceView();
    }
  });

  // Orientation tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.orientation = btn.dataset.orient;
      updateSliderLimits();
      updateSliceView();
    });
  });

  // Window preset buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.windowPreset = btn.dataset.window;

      const presetName = btn.dataset.window;
      if (presetName === "abdomen") displayWindow.textContent = "W: 400  L: 40";
      if (presetName === "soft_tissue") displayWindow.textContent = "W: 350  L: 50";
      if (presetName === "bone") displayWindow.textContent = "W: 1800 L: 400";
      if (presetName === "lung") displayWindow.textContent = "W: 1500 L: -600";

      updateSliceView();
    });
  });

  // Case Selector Change
  caseSelect.addEventListener("change", (e) => {
    state.currentCase = e.target.value;
    loadCaseMeta(state.currentCase);
  });

  // Toggle all organs
  btnToggleAll.addEventListener("click", () => {
    state.allSelected = !state.allSelected;
    if (state.allSelected) {
      state.activeOrgans = [1, 2, 3, 4, 6, 7, 8, 9, 11];
    } else {
      state.activeOrgans = [];
    }
    const checkboxes = organList.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach((cb) => (cb.checked = state.allSelected));
    updateSliceView();
  });

  // Upload Box
  uploadDropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleUpload(e.target.files[0]);
    }
  });

  // Drag and drop upload
  uploadDropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropZone.style.borderColor = "var(--accent-primary)";
  });
  uploadDropZone.addEventListener("dragleave", () => {
    uploadDropZone.style.borderColor = "var(--border-color)";
  });
  uploadDropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropZone.style.borderColor = "var(--border-color)";
    if (e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files[0]);
    }
  });

  // Inference Trigger
  btnRunInfer.addEventListener("click", () => {
    runInference(state.currentCase, false);
  });

  // Error banner dismiss
  btnDismissError.addEventListener("click", hideError);

  // How This Works modal
  btnHowItWorks.addEventListener("click", () => {
    howItWorksBackdrop.style.display = "flex";
  });
  btnCloseHowItWorks.addEventListener("click", () => {
    howItWorksBackdrop.style.display = "none";
  });
  howItWorksBackdrop.addEventListener("click", (e) => {
    if (e.target === howItWorksBackdrop) howItWorksBackdrop.style.display = "none";
  });
}

function handleUpload(file) {
  if (!file.name.endsWith(".nii") && !file.name.endsWith(".nii.gz")) {
    showError("Please upload a .nii or .nii.gz file.");
    return;
  }

  hideError();
  showSpinner(`Uploading ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)...`);
  const formData = new FormData();
  formData.append("file", file);

  fetch("/api/upload", {
    method: "POST",
    body: formData,
  })
    .then((r) => {
      if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || "Upload failed"); });
      return r.json();
    })
    .then((data) => {
      hideSpinner();
      state.currentCase = data.case_id;
      loadCases();
      loadCaseMeta(data.case_id);
      inferStatusText.textContent = `Uploaded ${file.name}. No ground truth available for uploads — segmentation only, no Dice score.`;
    })
    .catch((err) => {
      hideSpinner();
      showError("Upload failed: " + err.message);
    });
}

function runInference(caseId, confirmSlow) {
  hideError();
  btnRunInfer.disabled = true;
  const startedAt = Date.now();
  inferStatusText.textContent = "Running SuPreM inference…";
  showSpinner("Running real SuPreM UNet sliding-window inference…");

  const elapsedTimer = setInterval(() => {
    const secs = ((Date.now() - startedAt) / 1000).toFixed(1);
    spinnerText.textContent = `Running real SuPreM UNet sliding-window inference… (${secs}s elapsed)`;
    inferStatusText.textContent = `Running… ${secs}s elapsed`;
  }, 250);

  const url = `/api/case/${caseId}/infer` + (confirmSlow ? "?confirm_slow=true" : "");

  fetch(url, { method: "POST" })
    .then((r) => {
      if (!r.ok) return r.json().then((e) => { throw new Error(e.detail || "Inference failed"); });
      return r.json();
    })
    .then((data) => {
      clearInterval(elapsedTimer);

      if (data.status === "confirm_required") {
        hideSpinner();
        btnRunInfer.disabled = false;
        inferStatusText.textContent = "";
        const proceed = window.confirm(
          data.message + "\n\nRun inference on CPU now? This may take a while."
        );
        if (proceed) {
          runInference(caseId, true);
        }
        return;
      }

      hideSpinner();
      btnRunInfer.disabled = false;
      inferStatusText.textContent = `Done in ${data.elapsed_seconds}s on ${data.device_label}.`;
      loadCaseMeta(caseId);
    })
    .catch((err) => {
      clearInterval(elapsedTimer);
      hideSpinner();
      btnRunInfer.disabled = false;
      inferStatusText.textContent = "";
      showError("Inference failed: " + err.message);
    });
}

function showSpinner(text) {
  spinnerText.textContent = text;
  spinner.style.display = "flex";
}

function hideSpinner() {
  spinner.style.display = "none";
}
