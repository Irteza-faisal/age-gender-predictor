let model;
const video = document.getElementById('video');
const output = document.getElementById('output');
const startBtn = document.getElementById('startBtn');
const predictBtn = document.getElementById('predictBtn');
const loadingBarContainer = document.getElementById('loadingBarContainer');
const loadingBar = document.getElementById('loadingBar');

const cpuTimeDisplay = document.getElementById('cpuTime');
const memoryDisplay = document.getElementById('memoryUsage');
const storageDisplay = document.getElementById('storageUsage');

async function getModelSizeMB(url){
    let totalBytes = 0;

    // Fetch model.json
    const modelJson = await fetch(`${url}/model.json`);
    const model = await modelJson.json();

    // Add model.json size
    const modelJsonSize = parseInt(modelJson.headers.get('content-length') || '0');
    totalBytes += modelJsonSize;

    // Add all shard sizes
    for (const weightFile of model.weightsManifest[0].paths) {
        const weightResp = await fetch(`${url}/${weightFile}`, { method: 'HEAD' });
        const size = parseInt(weightResp.headers.get('content-length') || '0');
        totalBytes += size;
    }

    return totalBytes / (1024 * 1024); // MB
}

async function updatePerformanceStats(inferenceTimeMs) {
  cpuTimeDisplay.innerText = `Inference Time: ${inferenceTimeMs.toFixed(2)} ms`;

  // RAM usage (if supported)
  if (performance.memory) {
    const used = (performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(2);
    const total = (performance.memory.totalJSHeapSize / (1024 * 1024)).toFixed(2);
    memoryDisplay.innerText = `RAM Usage: ${used} MB / ${total} MB`;
  } else {
    memoryDisplay.innerText = 'RAM Usage: Not supported';
  }
}

startBtn.onclick = async () => {
    try {
        loadingBarContainer.style.display = 'block';
        console.log("Loading model...");

        const modelPath = '/tfjs_model';

        const sizeStart = performance.now();
        const modelSizeMB = await getModelSizeMB(modelPath);
        const sizeEnd = performance.now();

        const loadStart = performance.now();
        model = await tf.loadGraphModel(`${modelPath}/model.json`);
        const loadEnd = performance.now();

        const downloadTime = loadEnd - loadStart;

        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        predictBtn.style.display = 'inline-block';
        startBtn.disabled = true;
        loadingBarContainer.style.display = 'none';

        // Display model size and download time
        document.getElementById('storageUsage').innerText =
            `Model Size: ${modelSizeMB.toFixed(2)} MB`;

        document.getElementById('cpuTime').innerText =
            `Model Download Time: ${downloadTime.toFixed(2)} ms`;

    } 
    catch (err) {
        console.error("Camera or model error:", err);
        alert('Camera access failed. Check console for details.');
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
