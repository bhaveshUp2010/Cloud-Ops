const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const messageList = document.querySelector("#message-list");
const conversation = document.querySelector(".conversation");
const welcome = document.querySelector("#welcome");
const sendButton = document.querySelector("#send-button");
const fileInput = document.querySelector("#file-input");
const uploadTrigger = document.querySelector("#upload-trigger");
const uploadStatus = document.querySelector("#upload-status");
const threadStorageKey = "cloudops-sentinel-thread";

function createThreadId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `thread-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getThreadId() {
  try {
    let id = localStorage.getItem(threadStorageKey);
    if (!id) {
      id = createThreadId();
      localStorage.setItem(threadStorageKey, id);
    }
    return id;
  } catch {
    return createThreadId();
  }
}

let threadId = getThreadId();

function scrollToLatest() {
  window.requestAnimationFrame(() => {
    conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
  });
}

function addUserMessage(text) {
  const message = document.createElement("article");
  message.className = "message user";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  message.append(body);
  messageList.append(message);
  scrollToLatest();
}

function addAssistantFrame() {
  const message = document.createElement("article");
  message.className = "message assistant";
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none"><path d="M12 3.5 19 6v5.3c0 4.3-2.8 7.4-7 9.2-4.2-1.8-7-4.9-7-9.2V6l7-2.5Z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><path d="m9 12 2 2 4-4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "CloudOps Sentinel";
  const loading = document.createElement("div");
  loading.className = "loading-indicator";
  loading.setAttribute("aria-label", "Preparing a response");
  loading.innerHTML = "<span></span><span></span><span></span>";
  body.append(label, loading);
  message.append(avatar, body);
  messageList.append(message);
  scrollToLatest();
  return { body };
}

function appendResponseDetails(body, response) {
  const sources = Array.isArray(response.sources) ? response.sources : [];
  if (sources.length) {
    const list = document.createElement("div");
    list.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "Sources";
    list.append(title);

    sources.forEach((source) => {
      const parts = [source.title || source.source || "Reference"];
      if (source.page) parts.push(`page ${source.page}`);
      const label = parts.join(" · ");

      if (source.url) {
        try {
          const url = new URL(source.url, window.location.origin);
          if (url.protocol === "http:" || url.protocol === "https:") {
            const link = document.createElement("a");
            link.className = "source-link";
            link.href = url.href;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = label;
            list.append(link);
            return;
          }
        } catch {
          // Display malformed URLs as plain source labels.
        }
      }

      const item = document.createElement("span");
      item.className = "source-label";
      item.textContent = label;
      list.append(item);
    });
    body.append(list);
  }

  const details = [];
  if (response.used_web_search) details.push("Web-assisted");
  if (response.support_status) details.push(response.support_status);
  if (details.length) {
    const meta = document.createElement("div");
    meta.className = "answer-meta";
    details.forEach((detail) => {
      const chip = document.createElement("span");
      chip.className = "meta-chip";
      chip.textContent = detail;
      meta.append(chip);
    });
    body.append(meta);
  }
}

async function submitQuestion(question) {
  const normalized = question.trim();
  if (normalized.length < 2) {
    questionInput.focus();
    return;
  }

  if (welcome) welcome.hidden = true;
  addUserMessage(normalized);
  const assistant = addAssistantFrame();
  questionInput.value = "";
  questionInput.style.height = "auto";
  sendButton.disabled = true;
  const slowRequestTimer = window.setTimeout(() => {
    if (!assistant.body.querySelector(".loading-indicator")) return;
    const notice = document.createElement("div");
    notice.className = "slow-message";
    notice.textContent = "This is taking longer than usual. The assistant is still working.";
    assistant.body.append(notice);
    scrollToLatest();
  }, 20000);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: normalized, thread_id: threadId }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The assistant could not complete your request.");
    }
    if (data.thread_id) threadId = data.thread_id;

    const loading = assistant.body.querySelector(".loading-indicator");
    if (loading) loading.remove();
    const answer = document.createElement("div");
    answer.className = "answer-text";
    answer.textContent = data.answer || "No response was returned.";
    assistant.body.append(answer);
    appendResponseDetails(assistant.body, data);
  } catch (error) {
    const loading = assistant.body.querySelector(".loading-indicator");
    if (loading) loading.remove();
    const answer = document.createElement("div");
    answer.className = "answer-text";
    answer.textContent = error.message || "Unable to reach the assistant. Please try again.";
    assistant.body.append(answer);
  } finally {
    window.clearTimeout(slowRequestTimer);
    const notice = assistant.body.querySelector(".slow-message");
    if (notice) notice.remove();
    sendButton.disabled = false;
    questionInput.focus();
    scrollToLatest();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => submitQuestion(button.dataset.prompt || ""));
});

uploadTrigger.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  uploadStatus.classList.remove("error");
  uploadStatus.textContent = `Adding ${file.name}…`;
  uploadTrigger.disabled = true;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The document could not be added.");
    }
    uploadStatus.textContent = `${data.filename} added · ${data.chunks_indexed} sections indexed`;
  } catch (error) {
    uploadStatus.classList.add("error");
    uploadStatus.textContent = error.message || "Unable to add this document. Please try again.";
  } finally {
    uploadTrigger.disabled = false;
    fileInput.value = "";
  }
});
