/**
 * RAG Document Assistant - Frontend Application
 * Handles file uploads, chat interactions, and API communication
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const elements = {
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    documentsList: document.getElementById('documentsList'),
    docCount: document.getElementById('docCount'),
    statusIndicator: document.getElementById('statusIndicator'),
    chatContainer: document.getElementById('chatContainer'),
    messagesContainer: document.getElementById('messagesContainer'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    clearChatBtn: document.getElementById('clearChatBtn'),
    inputHint: document.getElementById('inputHint'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText'),
    toastContainer: document.getElementById('toastContainer'),
    providerSelect: document.getElementById('providerSelect')
};

// Application State
let state = {
    documents: [],
    chatHistory: [],
    isProcessing: false,
    currentProvider: 'gemini'  // Default provider
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    setupEventListeners();
    await checkServerStatus();
    await loadProviders();
    await loadDocuments();
    setupExampleQueries();
}

function setupEventListeners() {
    // File upload events
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadZone.addEventListener('dragover', handleDragOver);
    elements.uploadZone.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone.addEventListener('drop', handleDrop);

    // Chat events
    elements.sendBtn.addEventListener('click', handleSendMessage);
    elements.chatInput.addEventListener('keydown', handleInputKeydown);
    elements.chatInput.addEventListener('input', autoResizeTextarea);
    elements.clearChatBtn.addEventListener('click', clearChat);

    // Provider selection
    elements.providerSelect.addEventListener('change', handleProviderChange);
}

// ===== Provider Management =====
async function loadProviders() {
    try {
        const response = await fetch(`${API_BASE_URL}/providers`);
        if (response.ok) {
            const data = await response.json();
            state.currentProvider = data.current;
            elements.providerSelect.value = data.current;

            // Update options based on availability
            data.providers.forEach(provider => {
                const option = elements.providerSelect.querySelector(`option[value="${provider.name}"]`);
                if (option) {
                    option.textContent = provider.display_name;
                    option.disabled = !provider.available;
                    if (!provider.available) {
                        option.textContent += ' (Not configured)';
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error loading providers:', error);
    }
}

function handleProviderChange(e) {
    state.currentProvider = e.target.value;
    showToast('success', 'Provider Changed', `Now using ${e.target.options[e.target.selectedIndex].text}`);
}

function setupExampleQueries() {
    document.querySelectorAll('.example-query').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!btn.disabled && state.documents.length > 0) {
                elements.chatInput.value = btn.dataset.query;
                handleSendMessage();
            }
        });
    });
}

// ===== Server Status =====
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (response.ok) {
            updateStatus('connected', 'Connected');
            return true;
        }
    } catch (error) {
        updateStatus('error', 'Server unavailable');
        showToast('error', 'Connection Error', 'Cannot connect to the backend server. Make sure it is running.');
    }
    return false;
}

function updateStatus(status, text) {
    elements.statusIndicator.className = `status-indicator ${status}`;
    elements.statusIndicator.querySelector('span:last-child').textContent = text;
}

// ===== Document Management =====
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`);
        if (response.ok) {
            const oldDocs = JSON.stringify(state.documents);
            state.documents = await response.json();

            // If documents changed, re-render
            if (oldDocs !== JSON.stringify(state.documents)) {
                renderDocuments();
                updateInputState();
            }

            // Check if any documents are still "indexing" (0 chunks)
            const isIndexing = state.documents.some(doc => doc.total_chunks === 0);
            if (isIndexing) {
                // Poll every 3 seconds if indexing is in progress
                setTimeout(loadDocuments, 3000);
            }
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

function renderDocuments() {
    elements.docCount.textContent = state.documents.length;

    if (state.documents.length === 0) {
        elements.documentsList.innerHTML = `
            <div class="empty-state">
                <p>No documents uploaded yet</p>
            </div>
        `;
        return;
    }

    elements.documentsList.innerHTML = state.documents.map(doc => {
        const isIndexing = doc.total_chunks === 0;
        return `
            <div class="document-item ${isIndexing ? 'indexing' : ''}" data-id="${doc.id}">
                <div class="document-icon">
                    ${isIndexing
                ? `<div class="spinner-small"></div>`
                : `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                           </svg>`
            }
                </div>
                <div class="document-info">
                    <div class="document-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="document-meta">${isIndexing ? 'Indexing...' : `${doc.total_chunks} chunks indexed`}</div>
                </div>
                <button class="document-delete" onclick="deleteDocument('${doc.id}')" title="Remove document">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </button>
            </div>
        `;
    }).join('');
}

async function deleteDocument(docId) {
    try {
        const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast('success', 'Document Removed', 'The document has been removed from the index.');
            await loadDocuments();
        } else {
            throw new Error('Failed to delete document');
        }
    } catch (error) {
        showToast('error', 'Delete Failed', error.message);
    }
}

function updateInputState() {
    const hasDocuments = state.documents.length > 0;
    elements.chatInput.disabled = !hasDocuments;
    elements.sendBtn.disabled = !hasDocuments;

    if (hasDocuments) {
        elements.inputHint.textContent = 'Press Enter to send, Shift+Enter for new line';
        document.querySelectorAll('.example-query').forEach(btn => btn.disabled = false);
    } else {
        elements.inputHint.textContent = 'Upload documents to start asking questions';
        document.querySelectorAll('.example-query').forEach(btn => btn.disabled = true);
    }
}

// ===== File Upload =====
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    uploadFiles(files);
    e.target.value = ''; // Reset input
}

async function uploadFiles(files) {
    const validFiles = files.filter(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        return ['pdf', 'txt', 'md', 'text', 'docx'].includes(ext);
    });

    if (validFiles.length === 0) {
        showToast('warning', 'Invalid Files', 'Please upload PDF, TXT, DOCX, or MD files.');
        return;
    }

    // Don't show global loading overlay for background uploads
    // showLoading('Uploading documents...'); 

    for (const file of validFiles) {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                showToast('success', 'Upload Successful', result.message);
                // Immediately update list to show "Indexing..." status
                await loadDocuments();
            } else {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
        } catch (error) {
            showToast('error', 'Upload Failed', `${file.name}: ${error.message}`);
        }
    }
}

// ===== Chat Functionality =====
function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
}

function autoResizeTextarea() {
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 200) + 'px';
}

async function handleSendMessage() {
    const question = elements.chatInput.value.trim();

    if (!question || state.isProcessing) return;

    state.isProcessing = true;
    elements.chatInput.value = '';
    autoResizeTextarea();

    // Hide welcome message
    elements.welcomeMessage.style.display = 'none';

    // Add user message
    addMessage('user', question);

    // Add loading message
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                chat_history: state.chatHistory.slice(-6),
                provider: state.currentProvider
            })
        });

        if (response.ok) {
            const result = await response.json();

            // Remove loading message
            removeMessage(loadingId);

            // Add assistant response
            addMessage('assistant', result.answer, result.sources);

            // Update chat history
            state.chatHistory.push(
                { role: 'user', content: question },
                { role: 'assistant', content: result.answer }
            );
        } else {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get response');
        }
    } catch (error) {
        removeMessage(loadingId);
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
        showToast('error', 'Query Failed', error.message);
    }

    state.isProcessing = false;
    scrollToBottom();
}

function addMessage(role, content, sources = []) {
    const messageId = `msg-${Date.now()}`;
    const isUser = role === 'user';

    const avatarSvg = isUser
        ? `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2"/>
           </svg>`
        : `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
           </svg>`;

    const sourcesHtml = sources.length > 0 ? `
        <div class="message-sources">
            <div class="sources-header">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                </svg>
                Sources
            </div>
            ${sources.map(source => `
                <div class="source-item">
                    <div class="source-icon">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <span class="source-name">${source.filename}</span>
                    <span class="source-relevance">${source.relevance_score}% match</span>
                </div>
            `).join('')}
        </div>
    ` : '';

    const formattedContent = formatMessageContent(content);

    const messageHtml = `
        <div class="message ${role}" id="${messageId}">
            <div class="message-header">
                <div class="message-avatar">${avatarSvg}</div>
                <span class="message-sender">${isUser ? 'You' : 'DocAssist'}</span>
            </div>
            <div class="message-content">
                <div class="message-text">${formattedContent}</div>
                ${sourcesHtml}
            </div>
        </div>
    `;

    elements.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);
    scrollToBottom();

    return messageId;
}

function addLoadingMessage() {
    const messageId = `loading-${Date.now()}`;

    const messageHtml = `
        <div class="message assistant loading" id="${messageId}">
            <div class="message-header">
                <div class="message-avatar">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <span class="message-sender">DocAssist</span>
            </div>
            <div class="message-content">
                <div class="message-text">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <span>Thinking...</span>
                </div>
            </div>
        </div>
    `;

    elements.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);
    scrollToBottom();

    return messageId;
}

function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

function formatMessageContent(content) {
    // Convert markdown-like formatting to HTML
    return content
        .split('\n\n')
        .map(para => `<p>${para.replace(/\n/g, '<br>')}</p>`)
        .join('');
}

function clearChat() {
    state.chatHistory = [];
    elements.messagesContainer.innerHTML = '';
    elements.welcomeMessage.style.display = 'block';
}

function scrollToBottom() {
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

// ===== UI Utilities =====
function showLoading(text = 'Processing...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showToast(type, title, message) {
    const toastId = `toast-${Date.now()}`;

    const icons = {
        success: `<svg viewBox="0 0 24 24" fill="none"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="currentColor" stroke-width="2"/><path d="M22 4L12 14.01l-3-3" stroke="currentColor" stroke-width="2"/></svg>`,
        error: `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><path d="M15 9l-6 6M9 9l6 6" stroke="currentColor" stroke-width="2"/></svg>`,
        warning: `<svg viewBox="0 0 24 24" fill="none"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="currentColor" stroke-width="2"/><path d="M12 9v4M12 17h.01" stroke="currentColor" stroke-width="2"/></svg>`
    };

    const toastHtml = `
        <div class="toast ${type}" id="${toastId}">
            <div class="toast-icon">${icons[type]}</div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="removeToast('${toastId}')">
                <svg viewBox="0 0 24 24" fill="none"><path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/></svg>
            </button>
        </div>
    `;

    elements.toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    // Auto remove after 5 seconds
    setTimeout(() => removeToast(toastId), 5000);
}

function removeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.style.animation = 'slideUp 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }
}

// Make deleteDocument and removeToast available globally
window.deleteDocument = deleteDocument;
window.removeToast = removeToast;
