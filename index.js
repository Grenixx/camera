import {
    GestureRecognizer,
    FilesetResolver,
    DrawingUtils
  } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
  
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("output_canvas");
  const ctx = canvas.getContext("2d");
  const threeCanvas = document.getElementById("threeCanvas");
  
  const gestureNameEl = document.getElementById("gestureName");
  const handsCountEl = document.getElementById("handsCount");
  const timestampEl = document.getElementById("timestamp");
  
  let gestureRecognizer;
  let runningMode = "IMAGE";
  
  const renderer = new THREE.WebGLRenderer({ canvas: threeCanvas, alpha: true });
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;
  
  const geometry = new THREE.BoxGeometry(0.05, 0.05, 0.05);
  const material = new THREE.MeshNormalMaterial();
  const cube = new THREE.Mesh(geometry, material);
  scene.add(cube);
  
  let isLaunching = false;
  let launchVelocity = new THREE.Vector3();
  
  const resizeCanvases = () => {
    const width = video.videoWidth;
    const height = video.videoHeight;
    [canvas, threeCanvas].forEach(c => {
      c.width = width;
      c.height = height;
      c.style.width = width + "px";
      c.style.height = height + "px";
    });
    video.style.width = width + "px";
    video.style.height = height + "px";
    renderer.setSize(width, height);
  };
  
  const initGestureRecognizer = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
      },
      runningMode
    });
  };
  
  await initGestureRecognizer();
  
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        resizeCanvases();
        runDetection();
      };
    } catch (error) {
      console.error("Erreur lors de l'accès à la caméra : ", error);
    }
  };
  
  startWebcam();
  
  let lastVideoTime = -1;
  
  const runDetection = async () => {
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await gestureRecognizer.setOptions({ runningMode: "VIDEO" });
    }
  
    const drawingUtils = new DrawingUtils(ctx);
  
    const predict = async () => {
      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const nowInMs = Date.now();
        const results = await gestureRecognizer.recognizeForVideo(video, nowInMs);
  
        ctx.clearRect(0, 0, canvas.width, canvas.height);
  
        let gesture = "Aucun";
        let numHands = results.landmarks.length;
  
        if (results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
  
          drawingUtils.drawConnectors(
            landmarks,
            GestureRecognizer.HAND_CONNECTIONS,
            { color: "#00FF00", lineWidth: 3 }
          );
          drawingUtils.drawLandmarks(landmarks, {
            color: "#FF0000",
            lineWidth: 2
          });
  
          const palmIndices = [0, 1, 5, 9, 13, 17];
          let sumX = 0, sumY = 0, sumZ = 0;
          palmIndices.forEach(i => {
            sumX += landmarks[i].x;
            sumY += landmarks[i].y;
            sumZ += landmarks[i].z;
          });
          const avgX = sumX / palmIndices.length;
          const avgY = sumY / palmIndices.length;
          const avgZ = sumZ / palmIndices.length;
  
          const baseX = avgX * 2 - 1;
          const baseY = -(avgY * 2 - 1);
  
          const wrist = landmarks[0];
          const p5 = landmarks[5];
          const p17 = landmarks[17];
  
          const v1 = new THREE.Vector3(p5.x - wrist.x, p5.y - wrist.y, p5.z - wrist.z);
          const v2 = new THREE.Vector3(p17.x - wrist.x, p17.y - wrist.y, p17.z - wrist.z);
          const normal = new THREE.Vector3().crossVectors(v1, v2).normalize();
  
          const offset = 0.17;
          const cubeX = baseX + normal.x * offset;
          const cubeY = baseY - normal.y * offset;
          const cubeZ = normal.z * offset;
  
          if (!isLaunching) {
            cube.position.set(cubeX, cubeY, cubeZ);
          }
  
          cube.scale.set(5, 5, 5);
  
          const palmDir = v1.clone().normalize();
          const rotationMatrix = new THREE.Matrix4();
          rotationMatrix.lookAt(palmDir, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
          cube.rotation.setFromRotationMatrix(rotationMatrix);
        }
  
        if (results.gestures.length > 0) {
          gesture = results.gestures[0][0].categoryName;
  
          if (gesture === "Closed_Fist" && !isLaunching) {
            // Lancer le cube vers une direction (par exemple en haut à droite)
            isLaunching = true;
            launchVelocity.set(0.05, 0.05, 0);
          }
        }
  
        gestureNameEl.textContent = gesture;
        handsCountEl.textContent = numHands;
        timestampEl.textContent = new Date().toLocaleTimeString();
      }
  
      if (isLaunching) {
        cube.position.add(launchVelocity);
        // Réinitialiser s'il sort de l'écran
        if (cube.position.x > 1.5 || cube.position.y > 1.5) {
          isLaunching = false;
        }
      }
  
      renderer.render(scene, camera);
      requestAnimationFrame(predict);
    };
  
    predict();
  };
  