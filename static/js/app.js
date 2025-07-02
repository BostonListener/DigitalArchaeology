/**
 * Archaeological Discovery Pipeline - Frontend Application
 * 
 * Handles all client-side functionality including parameter editing,
 * real-time script execution monitoring, and WebSocket communications.
 */

class ArchaeologicalPipeline {
    constructor() {
        this.socket = null;
        this.autoScroll = true;
        this.currentScript = null;
        this.parameters = {};
        
        this.init();
    }
    
    init() {
        this.initializeWebSocket();
        this.bindEventListeners();
        this.loadParameters();
        this.updateStatus();
        
        console.log('Archaeological Pipeline Interface Initialized');
    }
    
    initializeWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.addConsoleMessage('Connected to pipeline server', 'success');
        });
        
        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.addConsoleMessage('Disconnected from pipeline server', 'warning');
        });
        
        this.socket.on('script_output', (data) => {
            this.addConsoleMessage(`[${data.script.toUpperCase()}] ${data.output}`, 'info');
        });
        
        this.socket.on('script_complete', (data) => {
            const status = data.success ? 'success' : 'error';
            const message = data.success ? 
                `${data.script.toUpperCase()} completed successfully` :
                `${data.script.toUpperCase()} failed (code: ${data.return_code})`;
            
            this.addConsoleMessage(message, status);
            this.onScriptComplete(data.script, data.success);
        });
        
        this.socket.on('script_error', (data) => {
            this.addConsoleMessage(`[${data.script.toUpperCase()}] ERROR: ${data.error}`, 'error');
            this.onScriptComplete(data.script, false);
        });
    }
    
    bindEventListeners() {
        // Parameter saving
        document.getElementById('save-config').addEventListener('click', () => {
            this.saveParameters();
        });
        
        // Script execution buttons
        document.getElementById('run-setup').addEventListener('click', () => {
            this.runScript('setup');
        });
        
        document.getElementById('run-pipeline').addEventListener('click', () => {
            this.runScript('pipeline');
        });
        
        document.getElementById('run-checkpoint').addEventListener('click', () => {
            this.runScript('checkpoint');
        });
        
        document.getElementById('run-visualization').addEventListener('click', () => {
            this.runScript('visualization');
        });
        
        // Script control
        document.getElementById('stop-script').addEventListener('click', () => {
            this.stopScript();
        });
        
        // Console controls
        document.getElementById('clear-console').addEventListener('click', () => {
            this.clearConsole();
        });
        
        document.getElementById('toggle-auto-scroll').addEventListener('click', () => {
            this.toggleAutoScroll();
        });
        
        // Parameter change detection
        this.bindParameterListeners();
    }
    
    bindParameterListeners() {
        const inputs = document.querySelectorAll('input[name]');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                this.markParametersChanged();
            });
        });
    }
    
    markParametersChanged() {
        const saveBtn = document.getElementById('save-config');
        saveBtn.innerHTML = '<i class="fas fa-save"></i> Save Configuration *';
        saveBtn.classList.add('btn-warning');
    }
    
    loadParameters() {
        fetch('/api/parameters')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    this.showNotification('No configuration found - using defaults', 'warning');
                    return;
                }
                
                this.parameters = data;
                this.populateParameterForm(data);
                this.showNotification('Configuration loaded successfully', 'success');
            })
            .catch(error => {
                console.error('Error loading parameters:', error);
                this.showNotification('Failed to load configuration', 'error');
            });
    }
    
    populateParameterForm(params) {
        const inputs = document.querySelectorAll('input[name]');
        inputs.forEach(input => {
            const path = input.name.split('.');
            let value = params;
            
            for (const key of path) {
                if (value && typeof value === 'object') {
                    value = value[key];
                } else {
                    value = undefined;
                    break;
                }
            }
            
            if (value !== undefined) {
                // Handle array values (convert back to comma-separated string)
                if (Array.isArray(value)) {
                    input.value = value.join(',');
                } else {
                    input.value = value;
                }
            }
        });
    }
    
    collectParametersFromForm() {
        const params = {};
        const inputs = document.querySelectorAll('input[name]');
        
        inputs.forEach(input => {
            const path = input.name.split('.');
            let value = input.value;
            
            // Handle array fields (comma-separated values)
            if (input.name.includes('parameter_grid') || input.name === 'sentinel_download.temporal_preference') {
                if (input.name === 'sentinel_download.temporal_preference') {
                    value = value.split(',').map(v => v.trim());
                } else {
                    value = value.split(',').map(v => parseFloat(v.trim()));
                }
            } else if (input.type === 'number') {
                value = parseFloat(value);
            }
            
            this.setNestedValue(params, path, value);
        });
        
        return params;
    }
    
    setNestedValue(obj, path, value) {
        const last = path.pop();
        const target = path.reduce((o, key) => {
            if (!(key in o)) o[key] = {};
            return o[key];
        }, obj);
        target[last] = value;
    }
    
    saveParameters() {
        const params = this.collectParametersFromForm();
        
        this.showLoading(true);
        
        fetch('/api/parameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            this.showLoading(false);
            
            if (data.success) {
                this.parameters = params;
                this.showNotification('Configuration saved successfully', 'success');
                
                // Reset save button
                const saveBtn = document.getElementById('save-config');
                saveBtn.innerHTML = '<i class="fas fa-save"></i> Save Configuration';
                saveBtn.classList.remove('btn-warning');
                
                this.addConsoleMessage('Configuration parameters updated', 'success');
            } else {
                this.showNotification(`Failed to save: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            this.showLoading(false);
            console.error('Error saving parameters:', error);
            this.showNotification('Failed to save configuration', 'error');
        });
    }
    
    runScript(scriptType) {
        if (this.currentScript) {
            this.showNotification('Another script is already running', 'warning');
            return;
        }
        
        this.currentScript = scriptType;
        this.updateScriptButtons(true);
        this.addConsoleMessage(`Starting ${scriptType} script...`, 'info');
        
        fetch('/api/run_script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ script_type: scriptType })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.showNotification(`${scriptType} script started`, 'success');
                this.addConsoleMessage(`Process ID: ${data.pid}`, 'info');
            } else {
                this.showNotification(`Failed to start ${scriptType}: ${data.error}`, 'error');
                this.onScriptComplete(scriptType, false);
            }
        })
        .catch(error => {
            console.error('Error running script:', error);
            this.showNotification(`Failed to start ${scriptType}`, 'error');
            this.onScriptComplete(scriptType, false);
        });
    }
    
    stopScript() {
        if (!this.currentScript) {
            this.showNotification('No script is currently running', 'warning');
            return;
        }
        
        fetch('/api/stop_script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.showNotification('Script stopped', 'success');
                this.addConsoleMessage('Script execution stopped by user', 'warning');
            } else {
                this.showNotification(`Failed to stop script: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            console.error('Error stopping script:', error);
            this.showNotification('Failed to stop script', 'error');
        });
    }
    
    onScriptComplete(scriptType, success) {
        this.currentScript = null;
        this.updateScriptButtons(false);
        this.updateStatus();
        
        if (success) {
            this.showNotification(`${scriptType} completed successfully!`, 'success');
        } else {
            this.showNotification(`${scriptType} failed - check console for details`, 'error');
        }
    }
    
    updateScriptButtons(running) {
        const actionButtons = document.querySelectorAll('.action-btn');
        const stopButton = document.getElementById('stop-script');
        
        actionButtons.forEach(btn => {
            btn.disabled = running;
        });
        
        stopButton.disabled = !running;
        
        // Update status indicator
        const statusElement = document.getElementById('pipeline-status');
        if (running) {
            statusElement.classList.add('running');
            statusElement.innerHTML = '<i class="fas fa-cog"></i> <span>Running...</span>';
        } else {
            statusElement.classList.remove('running');
            statusElement.innerHTML = '<i class="fas fa-cog"></i> <span>Ready</span>';
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (connected) {
            statusElement.classList.add('connected');
            statusElement.innerHTML = '<i class="fas fa-circle"></i> <span>Connected</span>';
        } else {
            statusElement.classList.remove('connected');
            statusElement.innerHTML = '<i class="fas fa-circle"></i> <span>Disconnected</span>';
        }
    }
    
    updateStatus() {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                // Update pipeline stage indicators
                this.updateStageStatus('stage1', data.stage1_complete);
                this.updateStageStatus('stage2', data.stage2_complete);
                this.updateStageStatus('stage3', data.stage3_complete);
                
                // Update script running status
                if (data.script_running && !this.currentScript) {
                    this.updateScriptButtons(true);
                } else if (!data.script_running && this.currentScript) {
                    this.updateScriptButtons(false);
                    this.currentScript = null;
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
            });
    }
    
    updateStageStatus(stage, complete) {
        // Add visual indicators for completed stages if needed
        // This could be enhanced with progress indicators
    }
    
    addConsoleMessage(message, type = 'info') {
        const console = document.getElementById('console-output');
        const timestamp = new Date().toLocaleTimeString();
        
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        
        line.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">${this.escapeHtml(message)}</span>
        `;
        
        console.appendChild(line);
        
        if (this.autoScroll) {
            console.scrollTop = console.scrollHeight;
        }
        
        // Limit console history to prevent memory issues
        const lines = console.querySelectorAll('.console-line');
        if (lines.length > 1000) {
            for (let i = 0; i < 100; i++) {
                if (lines[i]) {
                    lines[i].remove();
                }
            }
        }
    }
    
    clearConsole() {
        const console = document.getElementById('console-output');
        console.innerHTML = '';
        this.addConsoleMessage('Console cleared', 'info');
    }
    
    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const button = document.getElementById('toggle-auto-scroll');
        
        if (this.autoScroll) {
            button.classList.add('active');
            button.innerHTML = '<i class="fas fa-arrow-down"></i> Auto Scroll';
        } else {
            button.classList.remove('active');
            button.innerHTML = '<i class="fas fa-pause"></i> Manual Scroll';
        }
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Auto-remove notification after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
        
        // Allow manual removal by clicking
        notification.addEventListener('click', () => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        });
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.pipelineApp = new ArchaeologicalPipeline();
});

// Periodic status updates
setInterval(() => {
    if (window.pipelineApp) {
        window.pipelineApp.updateStatus();
    }
}, 10000); // Update every 10 seconds