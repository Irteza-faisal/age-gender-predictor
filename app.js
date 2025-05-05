let model;
const video = document.getElementById('video');
const output = document.getElementById('output');
const startBtn = document.getElementById('startBtn');
const predictBtn = document.getElementById('predictBtn');
const loadingBar = document.getElementById('loadingBar');
const statusMessage = document.getElementById('statusMessage');

const cpuTimeDisplay = document.getElementById('cpuTime');
const memoryDisplay = document.getElementById('memoryUsage');
const storageDisplay = document.getElementById('storageUsage');

function updateStatus(text, progressPercent) {
  statusMessage.innerText = text;
  loadingBar.style.width = `${progressPercent}%`;
}

async function getModelSizeMB(url) {
  let totalBytes = 0;
  const modelJson = await fetch(`${url}/model.json`);
  const model = await modelJson.json();
  totalBytes += parseInt(modelJson.headers.get('content-length') || '0');
  for (const weightFile of model.weightsManifest[0].paths) {
    const headResp = await fetch(`${url}/${weightFile}`, { method: 'HEAD' });
    totalBytes += parseInt(headResp.headers.get('content-length') || '0');
  }
  return totalBytes / (1024 * 1024); // MB
}

async function requestCameraAccess(retries = 2) {
  updateStatus("Requesting camera access...", 90);

  // Check browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera access is not supported in this browser.");
    updateStatus("Camera not supported.", 100);
    return;
  }

  // Check permission status, if available
  try {
    const permissionStatus = await navigator.permissions.query({ name: 'camera' });
    console.log("Camera permission state:", permissionStatus.state);

    if (permissionStatus.state === 'denied') {
      alert("Camera access has been permanently denied. Please allow it in your browser settings.");
      updateStatus("Camera permission denied.", 100);
      return;
    }
  } catch (err) {
    console.warn("Permission query not supported or failed:", err);
  }

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      console.log(`Camera access attempt ${attempt + 1}...`);

      // Stop previous stream if it exists
      if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
      }

      // Only request video (no audio needed)
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;

      updateStatus("Camera ready!", 100);
      console.log("Camera access granted and stream started.");
      return;

    } catch (err) {
      console.warn(`Attempt ${attempt + 1} failed:`, err);

      if (attempt === retries) {
        updateStatus("Failed to access camera. Please allow permission and refresh.", 100);
        alert(`Camera access failed: ${err.name} - ${err.message}`);
      }

      // Wait 1 second before retrying
      await new Promise(res => setTimeout(res, 1000));
    }
  }
}

startBtn.onclick = async () => {
  startBtn.disabled = true;
  predictBtn.style.display = 'none';
  updateStatus("Downloading model metadata...", 10);

  const modelPath = '/tfjs_model';

  try {
    const modelSizeMB = await getModelSizeMB(modelPath);
    storageDisplay.innerText = `Model Size: ${modelSizeMB.toFixed(2)} MB`;

    updateStatus("Loading model into memory...", 40);
    const loadStart = performance.now();
    model = await tf.loadGraphModel(`${modelPath}/model.json`);
    const loadEnd = performance.now();
    cpuTimeDisplay.innerText = `Model Download Time: ${(loadEnd - loadStart).toFixed(2)} ms`;

    updateStatus("Model loaded, initializing camera...", 70);
    await requestCameraAccess();

    predictBtn.style.display = 'inline-block';
  } catch (err) {
    console.error("Error during setup:", err);
    updateStatus("Failed to access camera. Please allow permission and refresh.", 100);
    alert('Camera access failed. Check your permissions.');
  }
};

predictBtn.onclick = async () => {
  const inputTensor = tf.tidy(() => {
    return tf.browser.fromPixels(video)
      .resizeNearestNeighbor([128, 128])
      .toFloat()
      .div(255.0)
      .expandDims();
  });

  const startTime = performance.now();
  const [ageTensor, genderTensor] = await model.predict(inputTensor);
  const endTime = performance.now();

  const age = (await ageTensor.data())[0];
  const genderProb = (await genderTensor.data())[0];
  const gender = genderProb > 0.5 ? 'Female' : 'Male';

  output.innerText = `Predicted Age: ${age.toFixed(1)}\nPredicted Gender: ${gender}`;
  tf.dispose([inputTensor, ageTensor, genderTensor]);

  await updatePerformanceStats(endTime - startTime);
};

async function updatePerformanceStats(inferenceTimeMs) {
  cpuTimeDisplay.innerText = `Inference Time: ${inferenceTimeMs.toFixed(2)} ms`;

  if (performance.memory) {
    const used = (performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(2);
    const total = (performance.memory.totalJSHeapSize / (1024 * 1024)).toFixed(2);
    memoryDisplay.innerText = `RAM Usage: ${used} MB / ${total} MB`;
  } else {
    memoryDisplay.innerText = 'RAM Usage: Not supported';
  }
}
