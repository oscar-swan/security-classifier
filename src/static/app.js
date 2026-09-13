const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const dropZoneText = document.getElementById("drop-zone-text");
const preview = document.getElementById("preview");
const analyseBtn = document.getElementById("analyse-btn");
const message = document.getElementById("message");
const results = document.getElementById("results");
const topLabel = document.getElementById("top-label");
const topConfidence = document.getElementById("top-confidence");
const bars = document.getElementById("bars");

let selectedFile = null;

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  analyseBtn.disabled = false;
  message.textContent = "";
  results.hidden = true;

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropZoneText.hidden = true;
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragover", "dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => e.preventDefault());
});

dropZone.addEventListener("dragover", () => dropZone.classList.add("is-dragover"));
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-dragover"));

dropZone.addEventListener("drop", (e) => {
  dropZone.classList.remove("is-dragover");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// Space/Enter on the focused label opens the file picker, matching click behaviour
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

analyseBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyseBtn.disabled = true;
  analyseBtn.textContent = "Analysing...";
  message.textContent = "";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      message.textContent = data.error || "Something went wrong.";
      results.hidden = true;
      return;
    }

    renderResults(data);
  } catch (err) {
    message.textContent = "Could not reach the server.";
  } finally {
    analyseBtn.disabled = false;
    analyseBtn.textContent = "Analyse";
  }
});

function renderResults(data) {
  topLabel.textContent = data.top_label;
  topConfidence.textContent = `${data.top_confidence}%`;

  bars.innerHTML = "";
  data.results.forEach(([className, confidence]) => {
    const li = document.createElement("li");
    li.className = "bar" + (className === data.top_label ? " is-top" : "");
    li.innerHTML = `
      <span>${className}</span>
      <span class="bar__track"><span class="bar__fill" style="width: ${confidence}%"></span></span>
      <span class="bar__value">${confidence}%</span>
    `;
    bars.appendChild(li);
  });

  results.hidden = false;
}