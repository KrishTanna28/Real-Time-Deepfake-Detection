// Background service worker for managing tab capture and communication
let activeDetectionTabId = null;

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'startDetection') {
    handleStartDetection(message.tabId, sendResponse);
    return true; // Keep channel open for async response
  } else if (message.action === 'stopDetection') {
    handleStopDetection(sendResponse);
    return true;
  } else if (message.action === 'captureTab') {
    handleCaptureTab(sender.tab.id, sendResponse);
    return true;
  } else if (message.action === 'detectionResult') {
    // Forward results to popup
    chrome.runtime.sendMessage(message);
  } else if (message.action === 'detectionError') {
    // Forward errors to popup
    chrome.runtime.sendMessage(message);
  } else if (message.action === 'detectionStopped') {
    activeDetectionTabId = null;
    chrome.storage.local.set({ isDetecting: false });
    chrome.runtime.sendMessage(message);
  }
});

// Start detection on a specific tab
async function handleStartDetection(tabId, sendResponse) {
  try {
    // Check if backend is available
    const settings = await chrome.storage.local.get(['backendUrl', 'captureInterval']);
    const backendUrl = settings.backendUrl || 'http://localhost:5000';
    const captureInterval = settings.captureInterval || 1000;

    // Test backend connection
    try {
      const response = await fetch(`${backendUrl}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error('Backend not responding');
      }
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: 'Backend server not available. Please start the backend server first.' 
      });
      return;
    }

    // Inject content script if not already injected
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['content.js']
      });
    } catch (error) {
      // Content script might already be injected, continue
      console.log('Content script already injected or error:', error);
    }

    // Send start message to content script
    chrome.tabs.sendMessage(tabId, {
      action: 'startDetection',
      interval: captureInterval
    }, (response) => {
      if (chrome.runtime.lastError) {
        sendResponse({ success: false, error: chrome.runtime.lastError.message });
      } else {
        activeDetectionTabId = tabId;
        chrome.storage.local.set({ isDetecting: true });
        sendResponse({ success: true });
      }
    });

  } catch (error) {
    console.error('Error starting detection:', error);
    sendResponse({ success: false, error: error.message });
  }
}

// Stop detection
function handleStopDetection(sendResponse) {
  if (activeDetectionTabId) {
    chrome.tabs.sendMessage(activeDetectionTabId, {
      action: 'stopDetection'
    }, (response) => {
      activeDetectionTabId = null;
      chrome.storage.local.set({ isDetecting: false });
      sendResponse({ success: true });
    });
  } else {
    chrome.storage.local.set({ isDetecting: false });
    sendResponse({ success: true });
  }
}

// Capture current tab as image
async function handleCaptureTab(tabId, sendResponse) {
  try {
    // Capture visible tab
    const dataUrl = await chrome.tabs.captureVisibleTab(null, {
      format: 'png',
      quality: 90
    });

    sendResponse({ dataUrl: dataUrl });
  } catch (error) {
    console.error('Error capturing tab:', error);
    sendResponse({ error: error.message });
  }
}

// Clean up when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === activeDetectionTabId) {
    activeDetectionTabId = null;
    chrome.storage.local.set({ isDetecting: false });
  }
});

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
  // Open popup (default behavior)
});

console.log('Deepfake Detection Extension: Background service worker loaded');
