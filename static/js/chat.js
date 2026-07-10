/* ════════════════════════════════════════════════════
   AdmitAI – Chat Page JavaScript
   Full ChatGPT-style chat interface with RAG support
════════════════════════════════════════════════════ */

"use strict";

/* ── State ───────────────────────────────────────── */
const state = {
    isLoading:       false,
    chatHistory:     [],
    lastAnswer:      "",
    collegeFilter:   "",
    msgCount:        0,
};

/* ── DOM refs ────────────────────────────────────── */
const el = {
    sidebar:        document.getElementById("sidebar"),
    sidebarToggle:  document.getElementById("sidebarToggle"),
    sidebarClose:   document.getElementById("sidebarClose"),
    sidebarOverlay: document.getElementById("sidebarOverlay"),
    newChatBtn:     document.getElementById("newChatBtn"),
    recentChats:    document.getElementById("recentChats"),
    collegeFilter:  document.getElementById("collegeFilter"),
    messagesInner:  document.getElementById("messagesInner"),
    messagesArea:   document.getElementById("messagesArea"),
    welcomeScreen:  document.getElementById("welcomeScreen"),
    questionInput:  document.getElementById("questionInput"),
    sendBtn:        document.getElementById("sendBtn"),
    downloadChatBtn:document.getElementById("downloadChatBtn"),
    clearChatBtn:   document.getElementById("clearChatBtn"),
    copyLastBtn:    document.getElementById("copyLastBtn"),
    statusBtn:      document.getElementById("statusBtn"),
    toastContainer: document.getElementById("toastContainer"),
    topbarSubtitle: document.getElementById("topbarSubtitle"),
};

/* ════════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════════ */
function openSidebar() {
    el.sidebar.classList.add("open");
    el.sidebarOverlay.classList.add("visible");
    document.body.style.overflow = "hidden";
}
function closeSidebar() {
    el.sidebar.classList.remove("open");
    el.sidebarOverlay.classList.remove("visible");
    document.body.style.overflow = "";
}

el.sidebarToggle?.addEventListener("click", () => {
    el.sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
});
el.sidebarClose?.addEventListener("click", closeSidebar);
el.sidebarOverlay?.addEventListener("click", closeSidebar);

/* ── College filter ──────────────────────────────── */
el.collegeFilter?.addEventListener("change", () => {
    state.collegeFilter = el.collegeFilter.value;
    const college = el.collegeFilter.options[el.collegeFilter.selectedIndex].text;
    if (el.topbarSubtitle) {
        el.topbarSubtitle.textContent = state.collegeFilter
            ? `Filtered: ${college}`
            : "Ask anything about admissions";
    }
});

/* ── Suggested questions ─────────────────────────── */
document.querySelectorAll(".suggestion-item, .suggestion-chip").forEach(item => {
    item.addEventListener("click", () => {
        const q = item.dataset.question;
        if (q && !state.isLoading) {
            el.questionInput.value = q;
            updateSendBtn();
            submitQuestion(q);
            closeSidebar();
        }
    });
});

/* ════════════════════════════════════════════════════
   INPUT HANDLING
════════════════════════════════════════════════════ */
function updateSendBtn() {
    const hasText = el.questionInput?.value.trim().length > 0;
    if (el.sendBtn) el.sendBtn.disabled = !hasText || state.isLoading;
}

el.questionInput?.addEventListener("input", () => {
    updateSendBtn();
    autoResize();
});

el.questionInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const q = el.questionInput.value.trim();
        if (q && !state.isLoading) submitQuestion(q);
    }
});

el.sendBtn?.addEventListener("click", () => {
    const q = el.questionInput?.value.trim();
    if (q && !state.isLoading) submitQuestion(q);
});

function autoResize() {
    const ta = el.questionInput;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
}

/* ════════════════════════════════════════════════════
   CORE: SUBMIT QUESTION
════════════════════════════════════════════════════ */
async function submitQuestion(question) {
    if (state.isLoading || !question.trim()) return;

    // Hide welcome screen
    if (el.welcomeScreen) el.welcomeScreen.style.display = "none";

    // Clear input
    el.questionInput.value = "";
    el.questionInput.style.height = "auto";
    updateSendBtn();

    // Add user message
    appendUserMessage(question);
    state.msgCount++;

    // Show typing indicator
    const typingId = showTypingIndicator();
    setLoading(true);

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                college_filter: state.collegeFilter || null,
            }),
        });

        removeTypingIndicator(typingId);

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) throw new Error(data.error);

        // Store in history
        state.lastAnswer = data.answer;
        state.chatHistory.push({ question, answer: data.answer, sources: data.sources });
        updateRecentChats();

        // Render AI message
        appendAssistantMessage(data.answer, data.sources, data.timestamp);

    } catch (err) {
        removeTypingIndicator(typingId);
        appendErrorMessage(err.message || "An unexpected error occurred.");
    } finally {
        setLoading(false);
        scrollToBottom();
    }
}

/* ════════════════════════════════════════════════════
   MESSAGE RENDERERS
════════════════════════════════════════════════════ */

function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
        <div class="message-meta" style="justify-content:flex-end">
            <span>${formatTime(new Date())}</span>
            <div class="avatar user-avatar">You</div>
        </div>
        <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    el.messagesInner.appendChild(row);
    scrollToBottom();
}

function appendAssistantMessage(text, sources, timestamp) {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.dataset.msgId = ++state.msgCount;

    const timeStr = timestamp ? formatTime(new Date(timestamp)) : formatTime(new Date());
    const renderedText = renderMarkdown(text);
    const sourcesHtml  = renderSources(sources);

    row.innerHTML = `
        <div class="message-meta">
            <div class="avatar ai-avatar">
                <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
                    <circle cx="14" cy="14" r="13" stroke="#0f62fe" stroke-width="2"/>
                    <path d="M8 18l4-8 3 5 2-3 3 6" stroke="#0f62fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <strong style="color:var(--text-primary)">AdmitAI</strong>
            <span>${timeStr}</span>
        </div>
        <div class="message-bubble">
            ${renderedText}
            ${sourcesHtml}
        </div>
        <div class="message-actions">
            <button class="msg-action-btn" onclick="copyMessage(this)">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" stroke-width="1.25"/><path d="M1.5 8.5V1.5h7" stroke="currentColor" stroke-width="1.25" stroke-linecap="round"/></svg>
                Copy
            </button>
        </div>
    `;
    el.messagesInner.appendChild(row);
}

function appendErrorMessage(errMsg) {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
        <div class="message-meta">
            <div class="avatar ai-avatar">⚠</div>
            <strong style="color:#f87171">Error</strong>
        </div>
        <div class="error-bubble">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style="flex-shrink:0;margin-top:2px"><circle cx="8" cy="8" r="7" stroke="#f87171" stroke-width="1.5"/><path d="M8 5v3M8 10v1" stroke="#f87171" stroke-width="1.5" stroke-linecap="round"/></svg>
            <span>${escapeHtml(errMsg)}</span>
        </div>
    `;
    el.messagesInner.appendChild(row);
}

/* ── Typing indicator ────────────────────────────── */
function showTypingIndicator() {
    const id = "typing-" + Date.now();
    const wrap = document.createElement("div");
    wrap.className = "message-row assistant";
    wrap.id = id;
    wrap.innerHTML = `
        <div class="message-meta">
            <div class="avatar ai-avatar">
                <svg width="14" height="14" viewBox="0 0 28 28" fill="none">
                    <circle cx="14" cy="14" r="13" stroke="#0f62fe" stroke-width="2"/>
                    <path d="M8 18l4-8 3 5 2-3 3 6" stroke="#0f62fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <strong style="color:var(--text-primary)">AdmitAI</strong>
            <span style="color:var(--ibm-blue-light);font-size:0.7rem">Thinking…</span>
        </div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    el.messagesInner.appendChild(wrap);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el2 = document.getElementById(id);
    if (el2) el2.remove();
}

/* ════════════════════════════════════════════════════
   MARKDOWN RENDERER (minimal, no external deps)
════════════════════════════════════════════════════ */
function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm,  "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm,   "<h1>$1</h1>");
    // Unordered lists
    html = html.replace(/^[•\-\*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");
    // Paragraphs
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    html = `<p>${html}</p>`;
    // Clean up empty p tags
    html = html.replace(/<p><\/p>/g, "");
    html = html.replace(/<p>(<[hulo])/g, "$1");
    html = html.replace(/(<\/[hulo][^>]*>)<\/p>/g, "$1");

    return html;
}

/* ── Sources HTML ────────────────────────────────── */
function renderSources(sources) {
    if (!sources || sources.length === 0) return "";

    const items = sources.map(s => `
        <div class="source-item">
            <span class="source-college">🏛 ${escapeHtml(s.college_name)}</span>
            <span class="source-doc">📄 ${escapeHtml(s.pdf_name)}</span>
            <span class="source-page">Page ${escapeHtml(String(s.page_number))}</span>
            <span class="source-type">${escapeHtml(s.document_type || "general")}</span>
        </div>
    `).join("");

    return `
        <div class="sources-card">
            <div class="sources-header">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><rect x="1" y="1" width="10" height="10" rx="2" stroke="#4589ff" stroke-width="1.5"/><path d="M3 4h6M3 6h4M3 8h5" stroke="#4589ff" stroke-width="1.5" stroke-linecap="round"/></svg>
                Sources (${sources.length})
            </div>
            ${items}
        </div>
    `;
}

/* ════════════════════════════════════════════════════
   ACTIONS
════════════════════════════════════════════════════ */

/* Copy last answer */
el.copyLastBtn?.addEventListener("click", async () => {
    if (!state.lastAnswer) { showToast("No answer to copy yet.", "info"); return; }
    await copyToClipboard(state.lastAnswer);
    showToast("Answer copied to clipboard!", "success");
});

/* Copy individual message */
window.copyMessage = async function (btn) {
    const bubble = btn.closest(".message-row").querySelector(".message-bubble");
    if (!bubble) return;
    // Strip HTML for clipboard
    const text = bubble.innerText || bubble.textContent;
    await copyToClipboard(text);
    showToast("Copied!", "success");
};

/* New chat */
el.newChatBtn?.addEventListener("click", () => {
    state.chatHistory = [];
    state.lastAnswer  = "";
    state.msgCount    = 0;
    el.messagesInner.innerHTML = "";

    const welcome = buildWelcomeScreen();
    el.messagesInner.appendChild(welcome);

    // Reattach chip handlers
    welcome.querySelectorAll(".suggestion-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            if (!state.isLoading) {
                submitQuestion(chip.dataset.question);
                closeSidebar();
            }
        });
    });

    updateRecentChats();
    closeSidebar();
    showToast("New conversation started.", "info");
});

function buildWelcomeScreen() {
    const div = document.createElement("div");
    div.className = "welcome-screen";
    div.innerHTML = `
        <div class="welcome-icon">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <circle cx="20" cy="20" r="18" stroke="#0f62fe" stroke-width="2"/>
                <path d="M10 28l6-12 5 8 4-5 5 9" stroke="#0f62fe" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <h2>How can I help you today?</h2>
        <p>Ask me anything about college admissions — fees, eligibility, scholarships, hostel, seat matrix, or placement.</p>
        <div class="suggested-questions">
            ${[
                "What is the fee structure for CSE at MANIT?",
                "How much is hostel fee at LNCT Bhopal?",
                "Which scholarships are available at IIT Indore?",
                "What documents are required for admission?",
                "Compare fee structure of IIT Indore and MANIT.",
                "What is the B.Tech CSE eligibility criteria?",
            ].map(q => `
                <button class="suggestion-chip" data-question="${escapeHtml(q)}">
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;opacity:0.6"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M5 5.5C5 4.67 5.67 4 6.5 4h1C8.33 4 9 4.67 9 5.5c0 .67-.4 1.24-1 1.48V8H6V6.48C5.4 6.24 5 5.67 5 5.5z" fill="currentColor"/><circle cx="7" cy="10" r="0.75" fill="currentColor"/></svg>
                    ${escapeHtml(q)}
                </button>
            `).join("")}
        </div>
    `;
    return div;
}

/* Clear chat via API */
el.clearChatBtn?.addEventListener("click", async () => {
    if (!confirm("Clear all chat history?")) return;
    try {
        await fetch("/api/clear-chat", { method: "POST" });
    } catch (_) { /* ignore */ }
    el.newChatBtn.click();
});

/* Download chat */
el.downloadChatBtn?.addEventListener("click", () => {
    if (!state.chatHistory.length) { showToast("No conversation to download.", "info"); return; }
    const lines = state.chatHistory.flatMap(entry => [
        `You: ${entry.question}`,
        `AdmitAI: ${entry.answer}`,
        entry.sources?.length
            ? "Sources:\n" + entry.sources.map(s =>
                `  • ${s.college_name} | ${s.pdf_name} | Page ${s.page_number}`
              ).join("\n")
            : "",
        "─".repeat(60),
    ]);
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = `admitai-chat-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
    showToast("Chat downloaded!", "success");
});

/* Status check */
el.statusBtn?.addEventListener("click", async () => {
    try {
        const res  = await fetch("/api/status");
        const data = await res.json();
        const vs   = data.vectorstore;
        const msg  = vs.status === "loaded"
            ? `✓ Vector DB: ${vs.total_vectors} vectors | ${data.colleges} colleges`
            : `⚠ Vector DB: ${vs.status}`;
        showToast(msg, vs.status === "loaded" ? "success" : "error");
    } catch (_) {
        showToast("Could not reach server.", "error");
    }
});

/* Update recent chats list */
function updateRecentChats() {
    if (!el.recentChats) return;
    el.recentChats.innerHTML = "";

    if (!state.chatHistory.length) {
        el.recentChats.innerHTML = `<div class="no-history">No recent chats yet</div>`;
        return;
    }

    [...state.chatHistory].reverse().slice(0, 12).forEach(entry => {
        const item = document.createElement("div");
        item.className = "chat-history-item";
        item.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;opacity:0.5">
                <path d="M7 1C3.69 1 1 3.69 1 7s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm1 9H6V6h2v4zm0-6H6V2h2v2z" fill="currentColor"/>
            </svg>
            <span>${escapeHtml(truncate(entry.question, 42))}</span>
        `;
        item.addEventListener("click", () => {
            if (!state.isLoading) {
                submitQuestion(entry.question);
                closeSidebar();
            }
        });
        el.recentChats.appendChild(item);
    });
}

/* ════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
════════════════════════════════════════════════════ */
function showToast(message, type = "info", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    el.toastContainer?.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        toast.style.transition = "opacity 0.3s, transform 0.3s";
        setTimeout(() => toast.remove(), 350);
    }, duration);
}

/* ════════════════════════════════════════════════════
   UTILITIES
════════════════════════════════════════════════════ */
function setLoading(loading) {
    state.isLoading = loading;
    if (el.sendBtn) {
        el.sendBtn.disabled = loading;
        el.sendBtn.classList.toggle("loading", loading);
    }
}

function scrollToBottom() {
    if (el.messagesArea) {
        el.messagesArea.scrollTo({ top: el.messagesArea.scrollHeight, behavior: "smooth" });
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(String(str)));
    return div.innerHTML;
}

function truncate(str, n) {
    return str.length > n ? str.substring(0, n - 1) + "…" : str;
}

function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch (_) {
        // Fallback for older browsers
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
    }
}

/* ── Init ────────────────────────────────────────── */
(function init() {
    updateSendBtn();
    updateRecentChats();
    el.questionInput?.focus();
})();
