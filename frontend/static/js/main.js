class HPGPTClient {
    constructor() {
        this.currentSessionId = null;
        this.websocket = null;
        this.isConnected = false;
        this.uploadedFiles = [];
        this.currentMessageDiv = null;
        this.currentMessageContent = null;
        this.isStreaming = false;
        this.typingQueue = [];
        this.isTyping = false;
        this.streamingStats = { chunks: 0, length: 0 };
        this.answerMode = 'specific';
        this.contextLimit = 10;
        this.currentSessionStats = { total_messages: 0, displayed_messages: 0 };

        
        
        // Smart scroll properties
        this.isUserScrolling = false;
        this.shouldAutoScroll = true;
        this.scrollTimeout = null;
        
        // TTS/STT properties
        this.speechSynthesis = window.speechSynthesis;
        this.speechRecognition = null;
        this.isListening = false;
        this.isSpeaking = false;
        this.ttsEnabled = false;
        this.sttEnabled = false;
        this.accumulatedSpeech = ''; // FIXED: Added missing property
        
        this.initializeElements();
        this.bindEvents();
        this.initializeSpeechFeatures();
        this.loadChatHistory();
        this.createNewSession();
        
        setTimeout(() => {
            this.createPersistentWatermark();
        }, 200);
    }
    
    initializeElements() {
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.fileInput = document.getElementById('file-input');
        this.fileUploadArea = document.getElementById('file-upload-area');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.chatHistory = document.getElementById('chat-history');
        this.agentSelector = document.getElementById('agent-type');
        this.chatTitle = document.getElementById('chat-title');
        this.answerModeToggle = document.getElementById('answer-mode-toggle');
        this.contextLimitInput = document.getElementById('context-limit');
    }
    
    // FIXED: Initialize Speech Features
    initializeSpeechFeatures() {
        // Initialize Speech Recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.speechRecognition = new SpeechRecognition();
            this.speechRecognition.continuous = false;
            this.speechRecognition.interimResults = true;
            this.speechRecognition.lang = 'en-US';
            
            this.speechRecognition.onstart = () => {
                this.isListening = true;
                this.updateMicrophoneButton();
                this.showListeningIndicator();
                console.log('🎤 Speech recognition started');
            };
            
            this.speechRecognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        transcript += event.results[i][0].transcript;
                    }
                }
                if (transcript) {
                    // FIXED: Append to accumulated speech instead of overwriting
                    this.accumulatedSpeech += (this.accumulatedSpeech ? ' ' : '') + transcript.trim();
                    this.messageInput.value = this.accumulatedSpeech;
                    this.messageInput.focus();
                }
            };
            
            this.speechRecognition.onend = () => {
                this.isListening = false;
                this.updateMicrophoneButton();
                this.hideListeningIndicator();
                console.log('🎤 Speech recognition ended');
                // Keep accumulated speech when recognition ends naturally
            };
            
            this.speechRecognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.isListening = false;
                this.updateMicrophoneButton();
                this.hideListeningIndicator();
                this.showSpeechError('Speech recognition failed. Please try again.');
            };
            
            this.sttEnabled = true;
        }
        
        // Check TTS availability
        if ('speechSynthesis' in window) {
            this.ttsEnabled = true;
            
            // Load voices when they become available
            if (speechSynthesis.onvoiceschanged !== undefined) {
                speechSynthesis.onvoiceschanged = () => {
                    console.log('🔊 TTS voices loaded');
                };
            }
        }
        
        // Add speech control buttons
        this.addSpeechControls();
    }
    
    // FIXED: Method to clear accumulated speech
    clearAccumulatedSpeech() {
        this.accumulatedSpeech = '';
        this.messageInput.value = '';
        console.log('🗑️ Speech input cleared');
    }
    
    // FIXED: Complete Add Speech Control Buttons
    addSpeechControls() {
        const inputArea = document.querySelector('.input-area');
        if (!inputArea) return;
        
        // Add microphone button for STT
        if (this.sttEnabled) {
            const micButton = document.createElement('button');
            micButton.id = 'mic-btn';
            micButton.className = 'speech-btn mic-btn';
            micButton.innerHTML = '🎤';
            micButton.title = 'Voice Input (Click and speak)';
            micButton.addEventListener('click', () => this.toggleSpeechRecognition());
            
            // Insert before send button
            inputArea.insertBefore(micButton, this.sendBtn);
            
            // Add clear speech button
            const clearBtn = document.createElement('button');
            clearBtn.id = 'clear-speech-btn';
            clearBtn.className = 'speech-btn clear-btn';
            clearBtn.innerHTML = '🗑️';
            clearBtn.title = 'Clear Voice Input';
            clearBtn.addEventListener('click', () => this.clearAccumulatedSpeech());
            
            // Insert before send button
            inputArea.insertBefore(clearBtn, this.sendBtn);
        }
    }
    
    // Toggle Speech Recognition
    toggleSpeechRecognition() {
        if (!this.speechRecognition) {
            this.showSpeechError('Speech recognition not supported in this browser');
            return;
        }
        
        if (this.isListening) {
            this.speechRecognition.stop();
        } else {
            try {
                this.speechRecognition.start();
            } catch (error) {
                console.error('Error starting speech recognition:', error);
                this.showSpeechError('Could not start voice input. Please check microphone permissions.');
            }
        }
    }
    
    // Update Microphone Button State
    updateMicrophoneButton() {
        const micBtn = document.getElementById('mic-btn');
        if (!micBtn) return;
        
        if (this.isListening) {
            micBtn.innerHTML = '🔴';
            micBtn.classList.add('listening');
            micBtn.title = 'Listening... Click to stop';
            micBtn.style.background = 'rgba(228, 0, 43, 0.2)';
            micBtn.style.animation = 'pulse 1.5s infinite';
        } else {
            micBtn.innerHTML = '🎤';
            micBtn.classList.remove('listening');
            micBtn.title = 'Voice Input (Click and speak)';
            micBtn.style.background = '';
            micBtn.style.animation = '';
        }
    }
    
    // Show Listening Indicator
    showListeningIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'listening-indicator';
        indicator.className = 'listening-indicator';
        indicator.innerHTML = `
            <div class="listening-animation">
                <span>🎤</span>
                <span>Listening...</span>
            </div>
        `;
        
        document.body.appendChild(indicator);
    }
    
    // Hide Listening Indicator
    hideListeningIndicator() {
        const indicator = document.getElementById('listening-indicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Show Speech Error
    showSpeechError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'speech-error';
        errorDiv.innerHTML = `<span>⚠️ ${message}</span>`;
        
        errorDiv.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: #dc2626;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (document.body.contains(errorDiv)) {
                    document.body.removeChild(errorDiv);
                }
            }, 300);
        }, 3000);
    }
    
    // ENHANCED: Speak Text with Better Voice Selection
    speakText(text, messageElement) {
        if (!this.ttsEnabled || !text) return;
        
        // Stop any current speech
        this.speechSynthesis.cancel();
        
        // Clean text for speech
        const cleanText = text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown
            .replace(/\*(.*?)\*/g, '$1') // Remove italic markdown
            .replace(/#{1,6}\s/g, '') // Remove headers
            .replace(/•/g, '') // Remove bullet points
            .replace(/`(.*?)`/g, '$1') // Remove code formatting
            .replace(/\[.*?\]/g, '') // Remove links
            .replace(/\n+/g, ' ') // Replace newlines with spaces
            .replace(/\s+/g, ' ') // Normalize spaces
            .trim();
        
        if (cleanText.length === 0) return;
        
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        
        // Select best available voice
        const voices = this.speechSynthesis.getVoices();
        const preferredVoice = voices.find(voice => 
            voice.lang.startsWith('en') && 
            (voice.name.includes('Natural') || 
             voice.name.includes('Neural') || 
             voice.name.includes('Premium') ||
             voice.name.includes('Enhanced'))
        ) || voices.find(voice => voice.lang.startsWith('en') && voice.localService) 
          || voices.find(voice => voice.lang.startsWith('en'));
        
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
        
        utterance.onstart = () => {
            this.isSpeaking = true;
            this.updateSpeakerButton(messageElement, true);
            console.log('🔊 TTS started');
        };
        
        utterance.onend = () => {
            this.isSpeaking = false;
            this.updateSpeakerButton(messageElement, false);
            console.log('🔊 TTS ended');
        };
        
        utterance.onerror = (event) => {
            console.error('TTS error:', event.error);
            this.isSpeaking = false;
            this.updateSpeakerButton(messageElement, false);
            this.showSpeechError('Text-to-speech failed. Please try again.');
        };
        
        this.speechSynthesis.speak(utterance);
    }
    
    // Update Speaker Button State
    updateSpeakerButton(messageElement, isSpeaking) {
        const speakerBtn = messageElement.querySelector('.speaker-btn');
        if (!speakerBtn) return;
        
        if (isSpeaking) {
            speakerBtn.innerHTML = '⏹️';
            speakerBtn.title = 'Stop Speaking';
            speakerBtn.classList.add('speaking');
            speakerBtn.style.background = 'rgba(34, 197, 94, 0.2)';
        } else {
            speakerBtn.innerHTML = '🔊';
            speakerBtn.title = 'Read Aloud';
            speakerBtn.classList.remove('speaking');
            speakerBtn.style.background = '';
        }
    }
    
    bindEvents() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.newChatBtn.addEventListener('click', () => this.createNewSession());
        
        this.answerModeToggle.addEventListener('change', (e) => {
            this.answerMode = e.target.checked ? 'detailed' : 'specific';
            console.log(`Answer mode changed to: ${this.answerMode}`);
            this.showModeChangeNotification();
        });
        
        this.contextLimitInput.addEventListener('change', (e) => {
            this.contextLimit = parseInt(e.target.value) || 10;
            console.log(`Context limit changed to: ${this.contextLimit}`);
            
            if (this.currentSessionId) {
                this.reloadSessionWithLimit();
            }
            
            this.showContextLimitNotification();
        });
        
        this.contextLimitInput.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            if (value < 1) {
                e.target.value = 1;
            } else if (value > 100) {
                e.target.value = 100;
            }
        });
        
        this.chatMessages.addEventListener('scroll', (e) => {
            this.handleUserScroll();
        });
    }
    
    handleUserScroll() {
        if (this.scrollTimeout) {
            clearTimeout(this.scrollTimeout);
        }
        
        const { scrollTop, scrollHeight, clientHeight } = this.chatMessages;
        const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
        
        if (isNearBottom) {
            this.shouldAutoScroll = true;
            this.isUserScrolling = false;
            this.hideScrollToBottomButton();
        } else {
            this.shouldAutoScroll = false;
            this.isUserScrolling = true;
            
            if (this.isStreaming) {
                this.showScrollToBottomButton();
            }
        }
        
        this.scrollTimeout = setTimeout(() => {
            this.isUserScrolling = false;
        }, 150);
    }
    
    scrollToBottom() {
        if (this.shouldAutoScroll && !this.isUserScrolling) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }
    
    forceScrollToBottom() {
        this.shouldAutoScroll = true;
        this.isUserScrolling = false;
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        this.hideScrollToBottomButton();
    }
    
    showScrollToBottomButton() {
        const existingBtn = document.getElementById('scroll-to-bottom-btn');
        if (existingBtn) {
            return;
        }
        
        const scrollBtn = document.createElement('div');
        scrollBtn.id = 'scroll-to-bottom-btn';
        scrollBtn.className = 'scroll-indicator';
        scrollBtn.innerHTML = '↓ New messages';
        
        scrollBtn.addEventListener('click', () => {
            this.forceScrollToBottom();
        });
        
        this.chatMessages.parentElement.appendChild(scrollBtn);
    }
    
    hideScrollToBottomButton() {
        const scrollBtn = document.getElementById('scroll-to-bottom-btn');
        if (scrollBtn) {
            scrollBtn.remove();
        }
    }
    
    showModeChangeNotification() {
        const notification = document.createElement('div');
        notification.className = 'mode-notification';
        notification.innerHTML = `
            <span>📝 Answer mode: <strong>${this.answerMode.charAt(0).toUpperCase() + this.answerMode.slice(1)}</strong></span>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: ${this.answerMode === 'detailed' ? '#e4002b' : '#00205b'};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 2000);
    }
    
    showContextLimitNotification() {
        const notification = document.createElement('div');
        notification.className = 'context-notification';
        notification.innerHTML = `
            <span>💬 Context limit: <strong>${this.contextLimit} messages</strong></span>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 120px;
            right: 20px;
            background: #00205b;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0, 32, 91, 0.3);
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 2000);
    }
    
    async reloadSessionWithLimit() {
        if (!this.currentSessionId) return;
        
        try {
            console.log(`Reloading session ${this.currentSessionId} with limit ${this.contextLimit}`);
            
            this.clearChat();
            
            const response = await fetch(`http://localhost:8000/sessions/${this.currentSessionId}/history?limit=${this.contextLimit}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Loaded limited chat history:', data);
            
            this.currentSessionStats = {
                total_messages: data.total_messages || 0,
                displayed_messages: data.displayed_messages || 0
            };
            
            if (data.history && data.history.length > 0) {
                this.displayLoadedHistory(data.history);
                
                if (data.limit_applied && data.total_messages > data.displayed_messages) {
                    this.showContextLimitInfo(data.displayed_messages, data.total_messages);
                }
            }
            
            this.updateContextInfo();
            
        } catch (error) {
            console.error('Error reloading session with limit:', error);
        }
    }
    
    updateContextInfo() {
        const contextContainer = this.contextLimitInput.parentElement;
        
        const existingInfo = contextContainer.querySelector('.context-info');
        if (existingInfo) {
            existingInfo.remove();
        }
        
        const contextInfo = document.createElement('div');
        contextInfo.className = 'context-info';
        contextInfo.innerHTML = `
            <small style="color: rgba(255, 255, 255, 0.8); font-size: 12px; margin-top: 4px; display: block;">
                Showing ${this.currentSessionStats.displayed_messages} of ${this.currentSessionStats.total_messages} messages
            </small>
        `;
        
        contextContainer.appendChild(contextInfo);
    }
    
    showContextLimitInfo(displayed, total) {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'context-limit-info';
        infoDiv.innerHTML = `
            <div style="
                background: rgba(0, 32, 91, 0.1);
                border: 1px solid rgba(0, 32, 91, 0.3);
                border-radius: 8px;
                padding: 12px;
                margin: 10px 0;
                text-align: center;
                font-size: 14px;
                color: #00205b;
            ">
                📝 Showing last <strong>${displayed}</strong> messages of <strong>${total}</strong> total messages.
                <br>
                <small>Increase context limit to see more messages.</small>
            </div>
        `;
        
        this.chatMessages.insertBefore(infoDiv, this.chatMessages.firstChild);
        
        setTimeout(() => {
            if (infoDiv.parentNode) {
                infoDiv.remove();
            }
        }, 5000);
    }
    
    createPersistentWatermark() {
        const existingWatermark = document.getElementById('persistent-watermark');
        if (existingWatermark) {
            existingWatermark.remove();
        }
        
        const watermarkDiv = document.createElement('div');
        watermarkDiv.id = 'persistent-watermark';
        watermarkDiv.className = 'persistent-watermark';
        
        const watermarkImg = document.createElement('img');
        watermarkImg.src = '/static/icons/logo.png';
        watermarkImg.alt = 'HPCL Watermark';
        watermarkImg.onerror = () => {
            console.log('Watermark image failed to load, trying alternative path');
            watermarkImg.src = './static/icons/logo.png';
        };
        
        watermarkDiv.appendChild(watermarkImg);
        document.body.appendChild(watermarkDiv);
        
        console.log('Fixed persistent watermark created');
    }
    
    async loadChatHistory() {
        try {
            console.log('Loading chat history...');
            const response = await fetch('http://localhost:8000/sessions');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Received sessions data:', data);
            
            this.displayChatHistory(data.sessions || []);
        } catch (error) {
            console.error('Error loading chat history:', error);
            this.chatHistory.innerHTML = '<div class="no-chats">Error loading chat history</div>';
        }
    }

    displayChatHistory(sessions) {
        console.log('Displaying sessions:', sessions);
        this.chatHistory.innerHTML = '';
        
        if (sessions.length === 0) {
            this.chatHistory.innerHTML = '<div class="no-chats">No previous chats</div>';
            return;
        }
        
        sessions.forEach(session => {
            console.log('Creating chat item for session:', session.session_id);
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.dataset.sessionId = session.session_id;
            
            if (session.session_id === this.currentSessionId) {
                chatItem.classList.add('active');
            }
            
            chatItem.innerHTML = `
                <div class="chat-item-content">
                    <div class="chat-title">${session.title}</div>
                    <div class="chat-meta">${this.formatDate(session.created_at)} • ${session.message_count} messages</div>
                </div>
                <button class="delete-chat-btn" data-session-id="${session.session_id}">
                    <img src="/static/icons/delete_bin.png" alt="Delete" class="delete-icon">
                </button>
            `;
            
            chatItem.addEventListener('click', (e) => {
                if (!e.target.closest('.delete-chat-btn')) {
                    this.loadSession(session.session_id, session.title);
                }
            });
            
            const deleteBtn = chatItem.querySelector('.delete-chat-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(session.session_id);
            });
            
            this.chatHistory.appendChild(chatItem);
        });
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 1) {
            return 'Today';
        } else if (diffDays === 2) {
            return 'Yesterday';
        } else if (diffDays <= 7) {
            return `${diffDays - 1} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    async createNewSession() {
        try {
            const response = await fetch('http://localhost:8000/sessions', {
                method: 'POST'
            });
            const data = await response.json();
            this.currentSessionId = data.session_id;
            this.chatTitle.textContent = 'New Chat';
            this.clearChat();
            this.connectWebSocket();
            this.loadChatHistory();
            
            console.log('New session created:', this.currentSessionId);
        } catch (error) {
            console.error('Error creating new session:', error);
        }
    }
    
    async loadSession(sessionId, title) {
        try {
            console.log(`Loading session: ${sessionId} - ${title}`);
            
            this.currentSessionId = sessionId;
            this.chatTitle.textContent = title;
            this.clearChat();
            
            const response = await fetch(`http://localhost:8000/sessions/${sessionId}/history?limit=${this.contextLimit}`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Loaded chat history with limit:', data);
            
            this.currentSessionStats = {
                total_messages: data.total_messages || 0,
                displayed_messages: data.displayed_messages || 0
            };
            
            if (data.history && data.history.length > 0) {
                this.displayLoadedHistory(data.history);
                
                if (data.limit_applied && data.total_messages > data.displayed_messages) {
                    this.showContextLimitInfo(data.displayed_messages, data.total_messages);
                }
            }
            
            this.connectWebSocket();
            this.updateActiveSession(sessionId);
            this.updateContextInfo();
            
        } catch (error) {
            console.error('Error loading session:', error);
            this.connectWebSocket();
            this.updateActiveSession(sessionId);
        }
    }

    displayLoadedHistory(history) {
        console.log('Displaying loaded history:', history);
        this.chatMessages.innerHTML = '';
        
        const originalAutoScroll = this.shouldAutoScroll;
        this.shouldAutoScroll = false;
        
        history.forEach((entry, entryIndex) => {
            console.log(`Processing history entry ${entryIndex}:`, entry);
            
            if (entry.messages && Array.isArray(entry.messages)) {
                entry.messages.forEach((msg, msgIndex) => {
                    console.log(`Adding message ${msgIndex}:`, msg.role, msg.content.substring(0, 50));
                    this.addMessage(msg.content, msg.role === 'user' ? 'user' : 'assistant', true);
                });
            }
        });
        
        setTimeout(() => {
            this.shouldAutoScroll = originalAutoScroll;
            this.forceScrollToBottom();
        }, 100);
        
        console.log('Finished displaying chat history');
    }

    updateActiveSession(sessionId) {
        this.chatHistory.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeItem = this.chatHistory.querySelector(`[data-session-id="${sessionId}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
    }
    
    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this chat? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`http://localhost:8000/sessions/${sessionId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                if (sessionId === this.currentSessionId) {
                    await this.createNewSession();
                } else {
                    this.loadChatHistory();
                }
                
                console.log('Session deleted successfully');
            } else {
                console.error('Failed to delete session');
            }
        } catch (error) {
            console.error('Error deleting session:', error);
        }
    }
    
    connectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        this.websocket = new WebSocket(`ws://localhost:8000/ws/${this.currentSessionId}`);
        
        this.websocket.onopen = () => {
            this.isConnected = true;
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            this.isConnected = false;
            console.log('WebSocket disconnected');
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleWebSocketMessage(data) {
        console.log('WebSocket message:', data.type, data.chunk_id || '');
        
        if (data.type === 'typing') {
            if (data.status === 'started') {
                this.showTypingIndicator();
            } else if (data.status === 'stopped') {
                this.removeTypingIndicator();
            }
        } else if (data.type === 'stream') {
            this.streamingStats.chunks++;
            this.streamingStats.length += data.content.length;
            
            this.typingQueue.push(data.content);
            
            if (!this.currentMessageDiv) {
                this.removeTypingIndicator();
                this.prepareAssistantMessage();
                this.startTypingAnimation();
            }
            
        } else if (data.type === 'complete') {
            console.log(`Streaming completed: ${data.total_chunks} chunks, ${data.total_length} characters`);
            
            this.waitForTypingCompletion().then(() => {
                this.finalizeCurrentMessage();
                this.removeTypingIndicator();
                this.isStreaming = false;
                this.enableSendButton();
                this.resetStreamingStats();
                this.loadChatHistory();
                
                this.hideScrollToBottomButton();
            });
            
        } else if (data.type === 'error') {
            this.removeTypingIndicator();
            this.addMessage(data.message || 'An error occurred. Please try again.', 'assistant');
            this.isStreaming = false;
            this.enableSendButton();
            this.resetStreamingStats();
        }
    }
    
    async waitForTypingCompletion() {
        while (this.typingQueue.length > 0 || this.isTyping) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }
    
    resetStreamingStats() {
        this.streamingStats = { chunks: 0, length: 0 };
    }
    
    async startTypingAnimation() {
        if (this.isTyping) return;
        this.isTyping = true;
        
        while (this.typingQueue.length > 0) {
            const content = this.typingQueue.shift();
            await this.typeContent(content);
        }
        
        this.isTyping = false;
    }
    
    async typeContent(content) {
        if (!this.currentMessageContent) return;
        
        for (let i = 0; i < content.length; i++) {
            if (!this.currentMessageContent) break;
            
            this.currentMessageContent.textContent += content[i];
            
            if (i % 3 === 0 || ['.', '!', '?', '\n', '*', '#', '•'].includes(content[i])) {
                this.applyRealTimeFormatting();
            }
            
            if (this.shouldAutoScroll && !this.isUserScrolling) {
                this.scrollToBottom();
            }
            
            let delay = 12;
            
            if (['.', '!', '?'].includes(content[i])) {
                delay = 120;
            } else if ([',', ';', ':'].includes(content[i])) {
                delay = 60;
            } else if (content[i] === ' ') {
                delay = 8;
            } else if (content[i] === '\n') {
                delay = 150;
            } else if (['*', '#', '•'].includes(content[i])) {
                delay = 30;
            } else if (content[i] === content[i].toUpperCase() && content[i] !== content[i].toLowerCase()) {
                delay = 15;
            }
            
            await new Promise(resolve => setTimeout(resolve, delay));
        }
        
        this.applyRealTimeFormatting();
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    applyRealTimeFormatting() {
        if (!this.currentMessageContent) return;
        
        const text = this.currentMessageContent.textContent;
        
        let formatted = text
            .replace(/(^|\n)### (.*?)(\n|$)/g, '$1<h3 style="margin: 15px 0 10px 0; color: #e4002b; font-size: 18px; font-weight: bold;">$2</h3>$3')
            .replace(/(^|\n)## (.*?)(\n|$)/g, '$1<h2 style="margin: 20px 0 15px 0; color: #00205b; font-size: 20px; font-weight: bold;">$2</h2>$3')
            .replace(/(^|\n)# (.*?)(\n|$)/g, '$1<h1 style="margin: 25px 0 20px 0; color: #e4002b; font-size: 24px; font-weight: bold;">$2</h1>$3')
            .replace(/\*\*([^*\n]+)\*\*/g, '<strong style="color: #00205b; font-weight: bold;">$1</strong>')
            .replace(/\*([^*\n]+)\*/g, '<em style="color: #666; font-style: italic;">$1</em>')
            .replace(/(^|\n)• (.*?)(\n|$)/g, '$1<div style="margin: 8px 0; padding-left: 20px; position: relative;"><span style="position: absolute; left: 0; color: #e4002b; font-weight: bold;">•</span>$2</div>$3')
            .replace(/`([^`\n]+)`/g, '<code style="background: rgba(0, 32, 91, 0.1); color: #00205b; padding: 2px 6px; border-radius: 4px; font-family: \'Courier New\', monospace; font-size: 14px;">$1</code>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>')
            .replace(/(^|\n)(\d+)\. (.*?)(\n|$)/g, '$1<div style="margin: 8px 0; padding-left: 25px; position: relative;"><span style="position: absolute; left: 0; color: #e4002b; font-weight: bold;">$2.</span>$3</div>$4');
        
        if (this.currentMessageContent.innerHTML !== formatted) {
            this.currentMessageContent.innerHTML = formatted;
        }
    }
    
    async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message && this.uploadedFiles.length === 0) return;

    if (this.isStreaming) return;

    this.disableSendButton();
    this.isStreaming = true;
    this.typingQueue = [];
    this.resetStreamingStats();

    if (message) {
        this.addMessage(message, 'user');
    }

    if (this.isConnected) {
        this.websocket.send(JSON.stringify({
            content: message,
            agent_type: this.agentSelector.value,
            answer_mode: this.answerMode,
            files: this.uploadedFiles
        }));
    }

    this.messageInput.value = '';
    this.uploadedFiles = [];
    this.updateFileUploadArea();

    this.clearAccumulatedSpeech();
}

    
    // FIXED: Add Message with Proper Button Placement
    addMessage(content, sender, isLoaded = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        if (isLoaded) {
            messageDiv.classList.add('loaded-message');
        }
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.formatMessage(content);
        
        messageDiv.appendChild(messageContent);
        
        // FIXED: Only add action buttons to assistant messages AFTER they're complete
        if (sender === 'assistant' && isLoaded) {
            this.addActionButtons(messageDiv, content);
        }
        
        this.chatMessages.appendChild(messageDiv);
        
        if (!isLoaded) {
            this.forceScrollToBottom();
        }
        
        return messageDiv;
    }
    
    // FIXED: Prepare Assistant Message WITHOUT buttons during typing
    prepareAssistantMessage() {
        this.currentMessageDiv = document.createElement('div');
        this.currentMessageDiv.className = 'message assistant typing-active';
        
        this.currentMessageContent = document.createElement('div');
        this.currentMessageContent.className = 'message-content';
        this.currentMessageContent.textContent = '';
        
        this.currentMessageDiv.appendChild(this.currentMessageContent);
        
        // NO ACTION BUTTONS DURING TYPING - they'll be added in finalizeCurrentMessage()
        
        this.chatMessages.appendChild(this.currentMessageDiv);
        this.forceScrollToBottom();
    }
    
    // FIXED: Finalize Message with Action Buttons at the END
    finalizeCurrentMessage() {
    if (this.currentMessageContent) {
        const rawText = this.currentMessageContent.textContent.trim();

        // Check if it's a code block (starts with triple backticks)
        const isCodeBlock = rawText.startsWith("```");
        let finalHTML;

        if (isCodeBlock) {
            // Strip outer triple backticks and optional language label
            const cleanedCode = rawText.replace(/^```[\w\+\#]*\s*\n/, '').replace(/```$/, '');
            finalHTML = `<pre><code>${this.escapeHtml(cleanedCode)}</code></pre>`;
        } else {
            // Normal text message
            finalHTML = this.escapeHtml(rawText).replace(/\n/g, "<br>");
        }

        this.currentMessageDiv.innerHTML = finalHTML;
        this.currentMessageDiv.classList.remove('typing-active');

        // Add action buttons after finalizing message
        this.addActionButtons(this.currentMessageDiv, rawText);

        console.log(`Message finalized: ${rawText.length} characters`);
    }

    this.currentMessageDiv = null;
    this.currentMessageContent = null;
}

    
    // Add Action Buttons with Speaker Button
    addActionButtons(messageDiv, messageContent) {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        
        // Speaker button for TTS
        if (this.ttsEnabled) {
            const speakerBtn = this.createSpeakerButton(messageContent, messageDiv);
            actionsDiv.appendChild(speakerBtn);
        }
        
        // Feedback buttons
        const thumbsUp = this.createActionButton('thumbs_up.png', 'thumbs-up', messageContent);
        const thumbsDown = this.createActionButton('thumbs_down.png', 'thumbs-down', messageContent);
        
        actionsDiv.appendChild(thumbsUp);
        actionsDiv.appendChild(thumbsDown);
        
        messageDiv.appendChild(actionsDiv);
    }
    
    // Create Speaker Button
    createSpeakerButton(messageContent, messageDiv) {
        const button = document.createElement('button');
        button.className = 'action-btn speaker-btn';
        button.innerHTML = '🔊';
        button.title = 'Read Aloud';
        button.addEventListener('click', () => {
            if (this.isSpeaking) {
                this.speechSynthesis.cancel();
                this.isSpeaking = false;
                this.updateSpeakerButton(messageDiv, false);
            } else {
                this.speakText(messageContent, messageDiv);
            }
        });
        return button;
    }
    
    createActionButton(iconName, action, messageContent) {
        const button = document.createElement('button');
        button.className = 'action-btn';
        button.innerHTML = `<img src="/static/icons/${iconName}" alt="${action}">`;
        button.addEventListener('click', () => this.handleFeedback(action, messageContent, button));
        return button;
    }
    
    async handleFeedback(action, messageContent, buttonElement) {
        console.log(`Feedback: ${action} for message: ${messageContent.substring(0, 100)}...`);
        
        const actionsDiv = buttonElement.parentElement;
        const allButtons = actionsDiv.querySelectorAll('.action-btn:not(.speaker-btn)');
        allButtons.forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
        });
        
        try {
            const feedbackData = {
                session_id: this.currentSessionId,
                message_content: messageContent,
                feedback_type: action === 'thumbs-up' ? 'positive' : 'negative',
                agent_type: this.agentSelector.value,
                answer_mode: this.answerMode,
                timestamp: new Date().toISOString()
            };
            
            const response = await fetch('http://localhost:8000/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('Feedback submitted successfully:', result);
                this.showFeedbackThankYou(actionsDiv, action);
            } else {
                throw new Error('Failed to submit feedback');
            }
            
        } catch (error) {
            console.error('Error submitting feedback:', error);
            
            allButtons.forEach(btn => {
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.style.cursor = 'pointer';
            });
            
            this.showFeedbackError(actionsDiv);
        }
    }
    
    showFeedbackThankYou(actionsDiv, action) {
        // Keep speaker button, replace feedback buttons
        const speakerBtn = actionsDiv.querySelector('.speaker-btn');
        
        const thankYouDiv = document.createElement('div');
        thankYouDiv.className = 'feedback-thank-you';
        
        const isPositive = action === 'thumbs-up';
        const icon = isPositive ? '👍' : '👎';
        const message = isPositive 
            ? 'Thanks! This helps me improve.' 
            : 'Thanks for the feedback! I\'ll work on improving.';
        
        thankYouDiv.innerHTML = `
            <span class="feedback-icon">${icon}</span>
            <span class="feedback-message">${message}</span>
        `;
        
        // Clear actions div and add speaker button + thank you
        actionsDiv.innerHTML = '';
        if (speakerBtn) {
            actionsDiv.appendChild(speakerBtn);
        }
        actionsDiv.appendChild(thankYouDiv);
        
        thankYouDiv.style.animation = 'fadeInUp 0.3s ease-out';
    }
    
    showFeedbackError(actionsDiv) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'feedback-error';
        errorDiv.innerHTML = `
            <span class="feedback-icon">⚠️</span>
            <span class="feedback-message">Failed to submit feedback. Please try again.</span>
        `;
        
        const originalHTML = actionsDiv.innerHTML;
        const errorContainer = document.createElement('div');
        errorContainer.className = 'feedback-error';
        errorContainer.innerHTML = errorDiv.innerHTML;
        
        actionsDiv.appendChild(errorContainer);
        
        setTimeout(() => {
            if (errorContainer.parentNode) {
                errorContainer.remove();
            }
        }, 3000);
    }
    
    showTypingIndicator() {
        // Don't show separate typing indicator anymore
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    disableSendButton() {
        this.sendBtn.disabled = true;
        this.sendBtn.textContent = 'Sending...';
    }
    
    enableSendButton() {
        this.sendBtn.disabled = false;
        this.sendBtn.textContent = 'Send';
    }
    
    async handleFileUpload(event) {
        const files = Array.from(event.target.files);
        
        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch(`http://localhost:8000/upload/${this.currentSessionId}`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    this.uploadedFiles.push({
                        name: file.name,
                        type: file.type,
                        content: result.content
                    });
                    this.updateFileUploadArea();
                }
            } catch (error) {
                console.error('File upload error:', error);
            }
        }
        
        event.target.value = '';
    }
    
    updateFileUploadArea() {
        this.fileUploadArea.innerHTML = '';
        
        this.uploadedFiles.forEach((file, index) => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'uploaded-file';
            fileDiv.innerHTML = `
                <span>${file.name}</span>
                <button class="remove-file" onclick="hpgpt.removeFile(${index})">×</button>
            `;
            this.fileUploadArea.appendChild(fileDiv);
        });
    }
    
    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.updateFileUploadArea();
    }
    escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    formatMessage(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/``````/g, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to HPGPT</h2>
                <p>Your AI assistant for HPCL. Upload documents, ask questions, or get help with analytics.</p>
                <div class="speech-features-info">
                    <p><strong>🎤 Voice Input:</strong> Click the microphone button to speak your questions</p>
                    <p><strong>🔊 Audio Responses:</strong> Click the speaker button to hear responses read aloud</p>
                    <p><strong>🗑️ Clear Voice:</strong> Click the trash button to clear accumulated voice input</p>
                </div>
            </div>
        `;
    }
}

const hpgpt = new HPGPTClient();
