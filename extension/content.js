// Content script for capturing tab content and displaying overlay
let overlayIframe = null;
let captureInterval = null;
let isCapturing = false;

// Create and inject overlay
function createOverlay() {
  if (overlayIframe) return;

  overlayIframe = document.createElement('iframe');
  overlayIframe.id = 'deepfake-detection-overlay';
  overlayIframe.src = chrome.runtime.getURL('overlay.html');
  overlayIframe.style.cssText = `
    position: fixed;
    top: 0;
    right: 0;
    width: 360px;
    height: 100vh;
    border: none;
    z-index: 999999;
    pointer-events: auto;
  `;
  
  document.body.appendChild(overlayIframe);
}

// Remove overlay
function removeOverlay() {
  if (overlayIframe) {
    overlayIframe.remove();
    overlayIframe = null;
  }
}

// Capture current tab as image
async function captureTab() {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ action: 'captureTab' }, (response) => {
      if (response && response.dataUrl) {
        resolve(response.dataUrl);
      } else {
        reject(new Error('Failed to capture tab'));
      }
    });
  });
}

// Send frame to backend for analysis
async function analyzeFrame(imageDataUrl) {
  try {
    // Get backend URL from storage
    const settings = await chrome.storage.local.get(['backendUrl']);
    const backendUrl = settings.backendUrl || 'http://localhost:5000';

    // Convert data URL to blob
    const response = await fetch(imageDataUrl);
    const blob = await response.blob();

    // Create form data
    const formData = new FormData();
    formData.append('frame', blob, 'frame.png');

    // Send to backend
    const analysisResponse = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      body: formData
    });

    if (!analysisResponse.ok) {
      throw new Error('Backend analysis failed');
    }

    const result = await analysisResponse.json();
    return result;
  } catch (error) {
    console.error('Error analyzing frame:', error);
    throw error;
  }
}

// Start capturing and analyzing
async function startDetection(interval = 1000) {
  if (isCapturing) return;

  isCapturing = true;
  createOverlay();

  // Update overlay with initial status
  updateOverlay({ status: 'analyzing' });

  captureInterval = setInterval(async () => {
    try {
      // Capture current tab
      const imageDataUrl = await captureTab();

      // Analyze frame
      const result = await analyzeFrame(imageDataUrl);

      // Update overlay with results
      updateOverlay(result);

      // Send results to popup
      chrome.runtime.sendMessage({
        action: 'detectionResult',
        data: result
      });

    } catch (error) {
      console.error('Detection error:', error);
      chrome.runtime.sendMessage({
        action: 'detectionError',
        error: error.message
      });
    }
  }, interval);
}

// Stop detection
function stopDetection() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  isCapturing = false;
  removeOverlay();

  chrome.runtime.sendMessage({ action: 'detectionStopped' });
}

// Update overlay with detection results
function updateOverlay(data) {
  if (!overlayIframe) return;

  overlayIframe.contentWindow.postMessage({
    type: 'updateResults',
    data: data
  }, '*');
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startDetection') {
    const interval = message.interval || 1000;
    startDetection(interval);
    sendResponse({ success: true });
  } else if (message.action === 'stopDetection') {
    stopDetection();
    sendResponse({ success: true });
  }
  return true; // Keep message channel open for async response
});

// Handle overlay messages
window.addEventListener('message', (event) => {
  if (event.data.type === 'overlayClose' || event.data.type === 'overlayStop') {
    stopDetection();
  }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  stopDetection();
});
