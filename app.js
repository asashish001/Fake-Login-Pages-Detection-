// static/app.js
const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const statusText = document.getElementById("status");
const resultBox = document.getElementById("resultBox");
const labelText = document.getElementById("labelText");
const phishProb = document.getElementById("phishProb");
const confText = document.getElementById("confText");
const bar = document.getElementById("bar");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    statusText.textContent = "Please select a PNG file.";
    return;
  }

  statusText.textContent = "Running model inference...";
  resultBox.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    const isPhishing = data.label === "Phishing";
    statusText.textContent = `Threshold: ${window.PRED_THRESHOLD}`;
    labelText.textContent = data.label;
    labelText.className = isPhishing ? "text-rose-400" : "text-emerald-300";
    phishProb.textContent = data.phishing_probability.toFixed(2);
    confText.textContent = data.confidence.toFixed(2);

    bar.style.width = `${data.confidence}%`;
    bar.className = `h-4 rounded-full transition-all duration-500 ${isPhishing ? "bg-rose-400" : "bg-emerald-300"}`;

    resultBox.classList.remove("hidden");
  } catch (err) {
    statusText.textContent = err.message;
  }
});
