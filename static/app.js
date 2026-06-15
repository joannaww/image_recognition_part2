const result = document.querySelector("#result");
const userList = document.querySelector("#userList");
const userCount = document.querySelector("#userCount");
const verifyUser = document.querySelector("#verifyUser");

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#tab-${button.dataset.tab}`).classList.add("active");
  });
});

document.querySelectorAll("input[type='file'][data-preview]").forEach((input) => {
  input.addEventListener("change", () => renderPreview(input));
});

bindForm("#enrollForm", "/api/enroll", renderEnrollResult);
bindForm("#verifyForm", "/api/verify", renderDecisionResult);
bindForm("#identifyForm", "/api/identify", renderIdentifyResult);

function bindForm(selector, url, renderer) {
  const form = document.querySelector(selector);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type='submit']");
    button.disabled = true;
    showBusy();

    try {
      const response = await fetch(url, { method: "POST", body: new FormData(form) });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        renderError(payload);
        return;
      }
      renderer(payload);
      if (payload.users) {
        renderUsers(payload.users);
      }
    } catch (error) {
      renderError({ message: error.message || "Nieznany błąd." });
    } finally {
      button.disabled = false;
    }
  });
}

function renderPreview(input) {
  const box = document.querySelector(`#${input.dataset.preview}`);
  box.innerHTML = "";
  Array.from(input.files).forEach((file) => {
    const item = document.createElement("div");
    item.className = "preview";
    const img = document.createElement("img");
    img.alt = file.name;
    img.src = URL.createObjectURL(file);
    img.addEventListener("load", () => URL.revokeObjectURL(img.src), { once: true });
    item.appendChild(img);
    box.appendChild(item);
  });
}

function showBusy() {
  result.className = "empty";
  result.textContent = "Przetwarzanie...";
}

function renderError(payload) {
  result.className = "result-card";
  result.innerHTML = `
    <span class="status bad">Błąd</span>
    <strong>${escapeHtml(payload.message || "Operacja nie powiodła się.")}</strong>
    ${renderRejected(payload.rejected || [])}
  `;
}

function renderEnrollResult(payload) {
  result.className = "result-card";
  result.innerHTML = `
    <span class="status good">Zapisano</span>
    <strong>${escapeHtml(payload.message)}</strong>
    <div class="accepted-list">
      ${(payload.accepted || []).map((item) => `
        <div class="file-row">
          <strong>${escapeHtml(item.file)}</strong>
        </div>
      `).join("")}
    </div>
    ${renderRejected(payload.rejected || [])}
  `;
}

function renderDecisionResult(payload) {
  const accepted = Boolean(payload.accepted);
  result.className = "result-card";
  result.innerHTML = `
    <span class="status ${accepted ? "good" : "bad"}">${accepted ? "Akceptacja" : "Odrzucenie"}</span>
    <strong>${escapeHtml(payload.user)}</strong>
    ${renderMetrics(payload)}
  `;
}

function renderIdentifyResult(payload) {
  const found = Boolean(payload.predicted_user);
  result.className = "result-card";
  result.innerHTML = `
    <span class="status ${found ? "good" : "warn"}">${found ? "Dopasowano" : "Open-set"}</span>
    <strong>${escapeHtml(payload.message)}</strong>
    ${renderMetrics(payload)}
    <div class="ranking">
      ${(payload.ranking || []).map((item, index) => `
        <div class="rank-row">
          <strong>${index + 1}. ${escapeHtml(item.user)}</strong>
          <span>${formatNumber(item.distance)}</span>
        </div>
      `).join("") || '<div class="empty">Brak rekordów w bazie.</div>'}
    </div>
  `;
}

function renderMetrics(payload) {
  return `
    <div class="metrics">
      ${metric("Dystans", payload.distance)}
      ${metric("Próg", payload.threshold)}
      ${metric("Jakość", payload.quality)}
    </div>
  `;
}

function metric(label, value) {
  return `
    <div class="metric">
      <span>${label}</span>
      <strong>${value === undefined ? "-" : formatNumber(value)}</strong>
    </div>
  `;
}

function renderRejected(items) {
  if (!items.length) {
    return "";
  }
  return `
    <div class="accepted-list">
      ${items.map((item) => `
        <div class="file-row">
          <strong>${escapeHtml(item.file)}</strong>
          <span>${escapeHtml(item.reason)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function renderUsers(users) {
  userCount.textContent = users.length;
  verifyUser.innerHTML = users
    .map((user) => `<option value="${escapeHtml(user.name)}">${escapeHtml(user.name)}</option>`)
    .join("");
  userList.innerHTML = users.length
    ? users.map((user) => `
      <div class="user-row">
        <strong>${escapeHtml(user.name)}</strong>
        <span>${user.embeddings} wzorc.</span>
      </div>
    `).join("")
    : '<div class="empty">Baza jest pusta.</div>';
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "-";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
