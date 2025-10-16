// Popup script for controlling the extension
let isDetecting = false;

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const resultsSection = document.getElementById('resultsSection');
const classificationEl = document.getElementById('classification');
const confidenceEl = document.getElementById('confidence');
const temporalAvgEl = document.getElementById('temporalAvg');
const temporalProgress = document.getElementById('temporalProgress');
const stabilityScoreEl = document.getElementById('stabilityScore');
const stabilityProgress = document.getElementById('stabilityProgress');
const framesAnalyzedEl = document.getElementById('framesAnalyzed');
const backendUrlInput = document.getElementById('backendUrl');
const captureIntervalInput = document.getElementById('captureInterval');
const testConnectionBtn = document.getElementById('testConnection');
const testContentScriptBtn = document.getElementById('testContentScript');
const connectionStatus = document.getElementById('connectionStatus');

// Load settings
chrome.storage.local.get(['backendUrl', 'captureInterval'], (result) => {
  if (result.backendUrl) {
    backendUrlInput.value = result.backendUrl;
  }
  if (result.captureInterval) {
    captureIntervalInput.value = result.captureInterval;
  }
});

// Auto-save settings on change
backendUrlInput.addEventListener('change', () => {
  chrome.storage.local.set({ backendUrl: backendUrlInput.value });
});

captureIntervalInput.addEventListener('change', () => {
  chrome.storage.local.set({ captureInterval: parseInt(captureIntervalInput.value) });
});

// Test backend connection
testConnectionBtn.addEventListener('click', async () => {
  const backendUrl = backendUrlInput.value || 'http://localhost:5000';
  connectionStatus.style.display = 'block';
  connectionStatus.textContent = '⏳ Testing backend...';
  connectionStatus.style.color = '#ffc107';
  
  try {
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET'
    });
    
    if (response.ok) {
      const data = await response.json();
      connectionStatus.textContent = `✅ Backend OK! Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'}, Device: ${data.device}`;
      connectionStatus.style.color = '#28a745';
    } else {
      connectionStatus.textContent = `❌ Backend error: ${response.status}`;
      connectionStatus.style.color = '#dc3545';
    }
  } catch (error) {
    connectionStatus.textContent = `❌ Backend failed: ${error.message}`;
    connectionStatus.style.color = '#dc3545';
  }
});

// Test content script injection
testContentScriptBtn.addEventListener('click', async () => {
  connectionStatus.style.display = 'block';
  connectionStatus.textContent = '⏳ Testing content script...';
  connectionStatus.style.color = '#ffc107';
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Check if we're on a valid page
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
      connectionStatus.textContent = `❌ Cannot run on chrome:// pages. Go to YouTube or any website.`;
      connectionStatus.style.color = '#dc3545';
      return;
    }
    
    // Try to ping content script (it's auto-injected)
    chrome.tabs.sendMessage(tab.id, { action: 'ping' }, (response) => {
      if (chrome.runtime.lastError) {
        connectionStatus.textContent = `❌ Content script not loaded. Refresh the page and try again.`;
        connectionStatus.style.color = '#dc3545';
      } else {
        connectionStatus.textContent = `✅ Content script OK! Ready to detect.`;
        connectionStatus.style.color = '#28a745';
      }
    });
  } catch (error) {
    connectionStatus.textContent = `❌ Test failed: ${error.message}`;
    connectionStatus.style.color = '#dc3545';
  }
});

// Start detection
startBtn.addEventListener('click', async () => {
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    console.log('Starting detection on tab:', tab.id);
    
    // Send message to background script to start detection
    chrome.runtime.sendMessage({
      action: 'startDetection',
      tabId: tab.id
    }, (response) => {
      console.log('Start detection response:', response);
      
      if (chrome.runtime.lastError) {
        console.error('Runtime error:', chrome.runtime.lastError);
        alert('Extension error: ' + chrome.runtime.lastError.message);
        return;
      }
      
      if (response && response.success) {
        isDetecting = true;
        updateUI();
        resultsSection.style.display = 'block';
      } else {
        const errorMsg = response?.error || 'Unknown error occurred';
        console.error('Detection failed:', errorMsg);
        alert('Failed to start detection.\n\n' + errorMsg + '\n\nMake sure:\n1. Backend server is running (python backend_server.py)\n2. Backend URL is correct (check settings)');
      }
    });
  } catch (error) {
    console.error('Error starting detection:', error);
    alert('Error: ' + error.message);
  }
});

// Stop detection
stopBtn.addEventListener('click', () => {
  chrome.runtime.sendMessage({ action: 'stopDetection' }, (response) => {
    if (response && response.success) {
      isDetecting = false;
      updateUI();
    }
  });
});

// Update UI based on detection state
function updateUI() {
  if (isDetecting) {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusDot.className = 'status-dot analyzing';
    statusText.textContent = 'Analyzing...';
  } else {
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusDot.className = 'status-dot';
    statusText.textContent = 'Inactive';
  }
}

// Listen for detection results
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'detectionResult') {
    updateResults(message.data);
  } else if (message.action === 'detectionError') {
    statusDot.className = 'status-dot alert';
    statusText.textContent = 'Error';
    console.error('Detection error:', message.error);
  } else if (message.action === 'detectionStopped') {
    isDetecting = false;
    updateUI();
  }
});

// Update results display
function updateResults(data) {
  if (!data) return;

  // Update classification
  const classification = data.confidence_level || 'UNCERTAIN';
  classificationEl.textContent = classification;
  classificationEl.className = 'result-value ' + classification.toLowerCase().replace('_', '-');

  // Update confidence
  const confidence = (data.fake_probability * 100).toFixed(1);
  confidenceEl.textContent = confidence + '%';

  // Update temporal average
  const temporalAvg = (data.temporal_average * 100).toFixed(1);
  temporalAvgEl.textContent = temporalAvg + '%';
  temporalProgress.style.width = temporalAvg + '%';

  // Update stability score
  const stability = (data.stability_score * 100).toFixed(1);
  stabilityScoreEl.textContent = stability + '%';
  stabilityProgress.style.width = stability + '%';

  // Update frames analyzed
  if (data.frame_count) {
    framesAnalyzedEl.textContent = data.frame_count;
  }

  // Update status based on classification
  if (classification === 'FAKE' || classification === 'HIGH_FAKE') {
    statusDot.className = 'status-dot alert';
    statusText.textContent = '🔴 Deepfake Detected!';
  } else if (classification === 'REAL' || classification === 'HIGH_REAL') {
    statusDot.className = 'status-dot active';
    statusText.textContent = '🟢 Authentic Video';
  } else {
    statusDot.className = 'status-dot analyzing';
    statusText.textContent = '🟡 Analyzing...';
  }
}

// Check if detection is already running
chrome.storage.local.get(['isDetecting'], (result) => {
  if (result.isDetecting) {
    isDetecting = true;
    updateUI();
    resultsSection.style.display = 'block';
  }
});
