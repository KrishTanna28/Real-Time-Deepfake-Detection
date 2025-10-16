// Script for overlay.html to handle updates and interactions
(function() {
  const statusBadge = document.getElementById('overlay-status');
  const classificationEl = document.getElementById('overlay-classification');
  const confidenceEl = document.getElementById('overlay-confidence');
  const temporalEl = document.getElementById('overlay-temporal');
  const stabilityEl = document.getElementById('overlay-stability');
  const framesEl = document.getElementById('overlay-frames');
  const closeBtn = document.getElementById('overlay-close');
  const stopBtn = document.getElementById('overlay-stop');

  // Handle close button
  closeBtn.addEventListener('click', () => {
    window.parent.postMessage({ type: 'overlayClose' }, '*');
  });

  // Handle stop button
  stopBtn.addEventListener('click', () => {
    window.parent.postMessage({ type: 'overlayStop' }, '*');
  });

  // Listen for updates from content script
  window.addEventListener('message', (event) => {
    if (event.data.type === 'updateResults') {
      updateDisplay(event.data.data);
    }
  });

  function updateDisplay(data) {
    if (!data) return;

    // Update status badge
    if (data.status) {
      statusBadge.className = 'status-badge ' + data.status;
      if (data.status === 'analyzing') {
        statusBadge.querySelector('.status-text').textContent = 'Analyzing...';
      }
    }

    // Update classification
    if (data.confidence_level) {
      const classification = data.confidence_level;
      classificationEl.textContent = classification;
      classificationEl.className = 'value ' + classification.toLowerCase().replace('_', '-');

      // Update status badge based on classification
      if (classification === 'HIGH_FAKE') {
        statusBadge.className = 'status-badge fake';
        statusBadge.querySelector('.status-text').textContent = 'Deepfake Detected!';
      } else if (classification === 'HIGH_REAL') {
        statusBadge.className = 'status-badge real';
        statusBadge.querySelector('.status-text').textContent = 'Authentic Video';
      } else {
        statusBadge.className = 'status-badge analyzing';
        statusBadge.querySelector('.status-text').textContent = 'Analyzing...';
      }
    }

    // Update confidence
    if (data.fake_probability !== undefined) {
      const confidence = (data.fake_probability * 100).toFixed(1);
      confidenceEl.textContent = confidence + '%';
    }

    // Update temporal average
    if (data.temporal_average !== undefined) {
      const temporal = (data.temporal_average * 100).toFixed(1);
      temporalEl.textContent = temporal + '%';
    }

    // Update stability score
    if (data.stability_score !== undefined) {
      const stability = (data.stability_score * 100).toFixed(1);
      stabilityEl.textContent = stability + '%';
    }

    // Update frames count
    if (data.frame_count !== undefined) {
      framesEl.textContent = data.frame_count;
    }
  }
})();
