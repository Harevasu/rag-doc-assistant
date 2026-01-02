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
    providerSelect: document.getElementById('providerSelect'),
    memoryToggle: document.getElementById('memoryToggle'),
    themeToggle: document.getElementById('themeToggle')
};

// Application State
let state = {
    documents: [],
    chatHistory: [],
    isProcessing: false,
    currentProvider: 'gemini',  // Default provider
    useMemory: true,  // Use conversation history
    indexingStatus: {},  // Track indexing progress per document
    currentTheme: 'dark'  // Default theme
};

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    loadTheme();
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

    // Memory toggle
    elements.memoryToggle.addEventListener('change', handleMemoryToggle);

    // Theme toggle
    elements.themeToggle.addEventListener('click', toggleTheme);
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

function handleMemoryToggle(e) {
    state.useMemory = e.target.checked;
    showToast('success', 'Memory ' + (state.useMemory ? 'Enabled' : 'Disabled'),
        state.useMemory ? 'Follow-up questions will use conversation history' : 'Each query is now isolated');
}

// ===== Theme Management =====
function loadTheme() {
    const savedTheme = localStorage.getItem('docassist-theme') || 'dark';
    state.currentTheme = savedTheme;
    applyTheme(savedTheme);
}

function toggleTheme() {
    const newTheme = state.currentTheme === 'dark' ? 'light' : 'dark';
    state.currentTheme = newTheme;
    applyTheme(newTheme);
    localStorage.setItem('docassist-theme', newTheme);
    showToast('success', 'Theme Changed', `Switched to ${newTheme} mode`);
}

function applyTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
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
            const indexingDocs = state.documents.filter(doc => doc.total_chunks === 0);
            if (indexingDocs.length > 0) {
                // Fetch detailed indexing status
                await fetchIndexingStatus();
                // Poll every 2 seconds if indexing is in progress
                setTimeout(loadDocuments, 2000);
            }
        }
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

async function fetchIndexingStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/documents/indexing`);
        if (response.ok) {
            const data = await response.json();
            const oldStatus = JSON.stringify(state.indexingStatus);
            state.indexingStatus = data.indexing_documents;
            
            // Update UI if status changed
            if (oldStatus !== JSON.stringify(state.indexingStatus)) {
                updateIndexingProgress();
            }
            
            // Check for newly completed documents
            for (const docId of Object.keys(state.indexingStatus)) {
                const status = state.indexingStatus[docId];
                if (status.status === 'completed') {
                    showToast('success', 'Indexing Complete', `${status.filename} is ready for queries!`);
                } else if (status.status === 'failed') {
                    showToast('error', 'Indexing Failed', `${status.filename}: ${status.error || 'Unknown error'}`);
                }
            }
        }
    } catch (error) {
        console.error('Error fetching indexing status:', error);
    }
}

function updateIndexingProgress() {
    // Update document items with detailed progress
    for (const [docId, status] of Object.entries(state.indexingStatus)) {
        const docItem = document.querySelector(`.document-item[data-id="${docId}"]`);
        if (docItem) {
            const metaEl = docItem.querySelector('.document-meta');
            if (metaEl && status.status === 'processing') {
                let progressText = status.current_step;
                if (status.estimated_chunks > 0) {
                    const percent = Math.round((status.chunks_processed / status.estimated_chunks) * 100);
                    progressText = `${status.current_step} (${percent}%)`;
                }
                if (status.eta_seconds && status.eta_seconds > 0) {
                    progressText += ` • ETA: ${formatTime(status.eta_seconds)}`;
                } else if (status.elapsed_seconds) {
                    progressText += ` • ${formatTime(status.elapsed_seconds)} elapsed`;
                }
                metaEl.textContent = progressText;
            }
        }
    }
}

function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
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
        const indexStatus = state.indexingStatus[doc.id];
        
        // Determine the status text
        let statusText = `${doc.total_chunks} chunks indexed`;
        let statusClass = '';
        
        if (isIndexing) {
            statusClass = 'indexing';
            if (indexStatus) {
                statusText = indexStatus.current_step || 'Processing...';
                if (indexStatus.estimated_chunks > 0 && indexStatus.chunks_processed > 0) {
                    const percent = Math.round((indexStatus.chunks_processed / indexStatus.estimated_chunks) * 100);
                    statusText += ` (${percent}%)`;
                }
            } else {
                statusText = 'Starting...';
            }
        }
        
        return `
            <div class="document-item ${statusClass}" data-id="${doc.id}">
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
                    <div class="document-meta">${statusText}</div>
                    ${isIndexing && indexStatus && indexStatus.elapsed_seconds ? 
                        `<div class="document-time">${formatTime(indexStatus.elapsed_seconds)} elapsed${indexStatus.eta_seconds ? ` • ETA: ${formatTime(indexStatus.eta_seconds)}` : ''}</div>` : ''}
                </div>
                <button class="document-delete" onclick="deleteDocument('${doc.id}')" title="Remove document" ${isIndexing ? 'disabled' : ''}>
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

    // Add streaming message placeholder
    const streamingId = addStreamingMessage();

    let fullAnswer = '';
    let sources = [];

    try {
        const response = await fetch(`${API_BASE_URL}/query/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                chat_history: state.useMemory ? state.chatHistory.slice(-6) : [],
                provider: state.currentProvider
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get streaming response');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'sources') {
                            sources = data.sources;
                        } else if (data.type === 'token') {
                            fullAnswer += data.content;
                            updateStreamingMessage(streamingId, fullAnswer);
                        } else if (data.type === 'done') {
                            // Finalize message with sources
                            finalizeStreamingMessage(streamingId, fullAnswer, sources);
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
        }

        // Update chat history
        state.chatHistory.push(
            { role: 'user', content: question },
            { role: 'assistant', content: fullAnswer }
        );

    } catch (error) {
        removeMessage(streamingId);
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
            ${sources.map(source => {
        const score = source.relevance_score;
        let levelClass, levelIcon, levelLabel;
        if (score >= 70) {
            levelClass = 'relevance-high';
            levelIcon = '🟢';
            levelLabel = 'High';
        } else if (score >= 40) {
            levelClass = 'relevance-medium';
            levelIcon = '🟡';
            levelLabel = 'Medium';
        } else {
            levelClass = 'relevance-low';
            levelIcon = '🔴';
            levelLabel = 'Low';
        }
        return `
                <div class="source-item" data-chunk="${encodeURIComponent(source.chunk_text || '')}" data-filename="${source.filename}" onclick="showSourceModal(this)">
                    <div class="source-icon">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <span class="source-name">${source.filename}</span>
                    <span class="source-relevance ${levelClass}" title="Relevance Score: ${score}%">
                        ${levelIcon} ${levelLabel} (${score}%)
                    </span>
                </div>
            `}).join('')}
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

function addStreamingMessage() {
    const messageId = `stream-${Date.now()}`;

    const messageHtml = `
        <div class="message assistant streaming" id="${messageId}">
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
                    <span class="thinking-text">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        Thinking...
                    </span>
                </div>
            </div>
        </div>
    `;

    elements.messagesContainer.insertAdjacentHTML('beforeend', messageHtml);
    scrollToBottom();

    return messageId;
}

function updateStreamingMessage(messageId, content) {
    const message = document.getElementById(messageId);
    if (message) {
        const textContainer = message.querySelector('.message-text');
        const formattedContent = formatMessageContent(content);
        textContainer.innerHTML = `
            <span class="streaming-text">${formattedContent}</span>
            <span class="streaming-cursor"></span>
        `;
        scrollToBottom();
    }
}

function finalizeStreamingMessage(messageId, content, sources) {
    const message = document.getElementById(messageId);
    if (message) {
        message.classList.remove('streaming');

        const formattedContent = formatMessageContent(content);
        const sourcesHtml = buildSourcesHtml(sources);

        const contentContainer = message.querySelector('.message-content');
        contentContainer.innerHTML = `
            <div class="message-text">${formattedContent}</div>
            ${sourcesHtml}
        `;
        scrollToBottom();
    }
}

function buildSourcesHtml(sources) {
    if (!sources || sources.length === 0) return '';

    return `
        <div class="message-sources">
            <div class="sources-header">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                </svg>
                Sources
            </div>
            ${sources.map(source => {
        const score = source.relevance_score;
        let levelClass, levelIcon, levelLabel;
        if (score >= 70) {
            levelClass = 'relevance-high';
            levelIcon = '🟢';
            levelLabel = 'High';
        } else if (score >= 40) {
            levelClass = 'relevance-medium';
            levelIcon = '🟡';
            levelLabel = 'Medium';
        } else {
            levelClass = 'relevance-low';
            levelIcon = '🔴';
            levelLabel = 'Low';
        }
        return `
                <div class="source-item" data-chunk="${encodeURIComponent(source.chunk_text || '')}" data-filename="${source.filename}" onclick="showSourceModal(this)">
                    <div class="source-icon">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2"/>
                        </svg>
                    </div>
                    <span class="source-name">${source.filename}</span>
                    <span class="source-relevance ${levelClass}" title="Relevance Score: ${score}%">
                        ${levelIcon} ${levelLabel} (${score}%)
                    </span>
                </div>
            `}).join('')}
        </div>
        ${getRelevanceWarning(sources)}
    `;
}

function getRelevanceWarning(sources) {
    const maxScore = Math.max(...sources.map(s => s.relevance_score || 0));

    if (maxScore < 40) {
        return `
            <div class="relevance-warning">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="currentColor" stroke-width="2"/>
                    <path d="M12 9v4M12 17h.01" stroke="currentColor" stroke-width="2"/>
                </svg>
                ⚠️ Answer may be weak due to low document relevance. Consider rephrasing your question.
            </div>
        `;
    }
    return '';
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

// ===== Source Modal =====
function showSourceModal(element) {
    const chunkText = decodeURIComponent(element.dataset.chunk || '');
    const filename = element.dataset.filename || 'Source Document';

    if (!chunkText) {
        showToast('warning', 'No Preview', 'Source text not available for this chunk.');
        return;
    }

    document.getElementById('sourceModalTitle').textContent = filename;
    document.getElementById('sourceChunkText').textContent = chunkText;
    document.getElementById('sourceModalOverlay').classList.add('active');
}

function closeSourceModal(event) {
    if (event && event.target !== document.getElementById('sourceModalOverlay')) {
        return;
    }
    document.getElementById('sourceModalOverlay').classList.remove('active');
}

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSourceModal();
    }
});

// Make deleteDocument, removeToast, and modal functions available globally
window.deleteDocument = deleteDocument;
window.removeToast = removeToast;
window.showSourceModal = showSourceModal;
window.closeSourceModal = closeSourceModal;
