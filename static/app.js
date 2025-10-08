document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const fileUpload = document.getElementById('file-upload');
    const fileSelected = document.getElementById('file-selected');
    const documentList = document.getElementById('document-list');
    const processBtn = document.getElementById('process-btn');
    const trainBtn = document.getElementById('train-btn');
    const resetBtn = document.getElementById('reset-btn');
    const statusMessage = document.getElementById('status-message');
    const promptForm = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    const chatMessages = document.getElementById('chat-messages');
    const modelStatus = document.getElementById('model-status');

    // Event Listeners
    fileUpload.addEventListener('change', updateFileSelection);
    uploadForm.addEventListener('submit', uploadDocuments);
    processBtn.addEventListener('click', processDocuments);
    trainBtn.addEventListener('click', trainModel);
    resetBtn.addEventListener('click', resetApplication);
    promptForm.addEventListener('submit', sendPrompt);

    // Check model status on load
    checkModelStatus();
    // Load document list on page load
    loadDocumentList();

    // Functions
    function updateFileSelection() {
        const numFiles = fileUpload.files.length;
        fileSelected.textContent = numFiles > 0 
            ? `${numFiles} file(s) selected` 
            : 'No files selected';
    }

    async function uploadDocuments(e) {
        e.preventDefault();
        
        if (fileUpload.files.length === 0) {
            addMessage('Please select at least one file to upload.', 'system');
            return;
        }

        const formData = new FormData();
        for (const file of fileUpload.files) {
            formData.append('files', file);
        }

        try {
            addMessage('Uploading documents...', 'system');
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                addMessage(`Successfully uploaded ${data.files.length} document(s).`, 'system');
                loadDocumentList();
                fileUpload.value = '';
                updateFileSelection();
            } else {
                addMessage(`Error: ${data.message || 'Failed to upload documents'}`, 'system');
            }
        } catch (error) {
            addMessage(`Error: ${error.message}`, 'system');
        }
    }

    async function processDocuments() {
        try {
            addMessage('Processing documents...', 'system');
            showStatusMessage('Processing documents...', 'info');
            processBtn.disabled = true;
            
            const response = await fetch('/process', {
                method: 'POST'
            });

            const data = await response.json();
            
            if (response.ok) {
                addMessage(data.message, 'system');
                showStatusMessage('Documents processed successfully!', 'success');
            } else {
                addMessage(`Error: ${data.message || 'Failed to process documents'}`, 'system');
                showStatusMessage('Failed to process documents', 'error');
            }
        } catch (error) {
            addMessage(`Error: ${error.message}`, 'system');
            showStatusMessage('Error processing documents', 'error');
        } finally {
            processBtn.disabled = false;
        }
    }

    async function trainModel() {
        try {
            addMessage('Training model on documents...', 'system');
            showStatusMessage('Training model on documents...', 'info');
            trainBtn.disabled = true;
            
            const response = await fetch('/train', {
                method: 'POST'
            });

            const data = await response.json();
            
            if (response.ok) {
                addMessage(data.message, 'system');
                showStatusMessage('Model training completed!', 'success');
            } else {
                addMessage(`Error: ${data.message || 'Failed to train model'}`, 'system');
                showStatusMessage('Failed to train model', 'error');
            }
        } catch (error) {
            addMessage(`Error: ${error.message}`, 'system');
            showStatusMessage('Error training model', 'error');
        } finally {
            trainBtn.disabled = false;
        }
    }

    async function sendPrompt(e) {
        e.preventDefault();
        
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Add user message to chat
        addMessage(prompt, 'user');
        
        // Clear input
        promptInput.value = '';

        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            
            const response = await fetch('/query', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                // Add AI response to chat
                addMessage(data.response, 'ai');
                
                // Add sources if available
                if (data.sources && data.sources.length > 0) {
                    const sourcesList = data.sources.join(', ');
                    addMessage(`Sources: ${sourcesList}`, 'system');
                }
            } else {
                addMessage(`Error: ${data.message || 'Failed to get response'}`, 'system');
            }
        } catch (error) {
            addMessage(`Error: ${error.message}`, 'system');
        }
    }

    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messagePara = document.createElement('p');
        messagePara.textContent = text;
        
        messageDiv.appendChild(messagePara);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function loadDocumentList() {
        try {
            // This would be a real endpoint in a production app
            // For now, we'll just check the documents directory
            const files = await fetch('/documents')
                .then(response => response.json())
                .catch(() => {
                    // If endpoint doesn't exist, just show a placeholder
                    return { files: [] };
                });
            
            documentList.innerHTML = '';
            
            if (files.files && files.files.length > 0) {
                files.files.forEach(file => {
                    const li = document.createElement('li');
                    li.textContent = file;
                    documentList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.textContent = 'No documents uploaded yet';
                documentList.appendChild(li);
            }
        } catch (error) {
            console.error('Error loading document list:', error);
        }
    }

    async function resetApplication() {
        if (!confirm('Are you sure you want to reset the application? This will delete all uploaded documents and reset the vector store.')) {
            return;
        }
        
        try {
            addMessage('Resetting application...', 'system');
            showStatusMessage('Resetting application...', 'info');
            resetBtn.disabled = true;
            
            const response = await fetch('/reset', {
                method: 'POST'
            });

            const data = await response.json();
            
            if (response.ok) {
                addMessage(data.message, 'system');
                showStatusMessage('Application reset successfully!', 'success');
                loadDocumentList(); // Refresh the document list
            } else {
                addMessage(`Error: ${data.message || 'Failed to reset application'}`, 'system');
                showStatusMessage('Failed to reset application', 'error');
            }
        } catch (error) {
            addMessage(`Error: ${error.message}`, 'system');
            showStatusMessage('Error resetting application', 'error');
        } finally {
            resetBtn.disabled = false;
        }
    }
    
    function showStatusMessage(message, type) {
        statusMessage.textContent = message;
        statusMessage.className = 'status-message';
        
        if (type) {
            statusMessage.classList.add(type);
        }
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
    }
    
    async function checkModelStatus() {
        try {
            // This would check if the model is loaded correctly
            // For now, we'll just update the status with a placeholder
            modelStatus.textContent = 'Local LLM ready for use. Upload documents to get started.';
        } catch (error) {
            modelStatus.textContent = 'Error loading model. Check console for details.';
            console.error('Error checking model status:', error);
        }
    }
});