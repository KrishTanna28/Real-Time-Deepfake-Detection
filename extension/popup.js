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
const saveSettingsBtn = document.getElementById('saveSettings');

// Load settings
chrome.storage.local.get(['backendUrl', 'captureInterval'], (result) => {
  if (result.backendUrl) {
    backendUrlInput.value = result.backendUrl;
  }
  if (result.captureInterval) {
    captureIntervalInput.value = result.captureInterval;
  }
});

// Save settings
saveSettingsBtn.addEventListener('click', () => {
  const settings = {
    backendUrl: backendUrlInput.value,
    captureInterval: parseInt(captureIntervalInput.value)
  };
  
  chrome.storage.local.set(settings, () => {
    // Visual feedback
    saveSettingsBtn.textContent = '✓ Saved';
    setTimeout(() => {
      saveSettingsBtn.textContent = 'Save Settings';
    }, 1500);
  });
});

// Start detection
startBtn.addEventListener('click', async () => {
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Send message to background script to start detection
    chrome.runtime.sendMessage({
      action: 'startDetection',
      tabId: tab.id
    }, (response) => {
      if (response && response.success) {
        isDetecting = true;
        updateUI();
        resultsSection.style.display = 'block';
      } else {
        alert('Failed to start detection. Make sure the backend server is running.');
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
  if (classification === 'HIGH_FAKE') {
    statusDot.className = 'status-dot alert';
    statusText.textContent = 'Deepfake Detected!';
  } else if (classification === 'HIGH_REAL') {
    statusDot.className = 'status-dot active';
    statusText.textContent = 'Authentic Video';
  } else {
    statusDot.className = 'status-dot analyzing';
    statusText.textContent = 'Analyzing...';
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
