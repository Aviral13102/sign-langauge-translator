/**
 * translator.js -- WebRTC & Client-side UI Logic
 * Handles webcam capture, drawing landmarks, UI updates, and TTS.
 */

// ---- DOM Elements ----
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const fpsCounter = document.getElementById('fps-counter');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const noHandOverlay = document.getElementById('no-hand-overlay');

const predLetter = document.getElementById('pred-letter');
const predConfidence = document.getElementById('pred-confidence');
const predictionBadge = document.getElementById('prediction-badge');

const bigLetter = document.getElementById('big-letter');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceLabel = document.getElementById('confidence-label');

const holdPanel = document.getElementById('hold-panel');
const holdBar = document.getElementById('hold-bar');
const holdLabel = document.getElementById('hold-label');

const wordText = document.getElementById('word-text');
const historyText = document.getElementById('history-text');

// ---- Controls ----
const btnSpeak = document.getElementById('btn-speak');
const btnClear = document.getElementById('btn-clear');
const btnBackspace = document.getElementById('btn-backspace');
const btnReset = document.getElementById('btn-reset');
const btnTtsToggle = document.getElementById('btn-tts-toggle');

// ---- State variables ----
let isRunning = false;
let ttsEnabled = true;
let currentWord = "";
let historyContent = "";

// Word Builder Config
const BUFFER_SIZE = 12;
const HOLD_FRAMES_REQUIRED = 15;
let predictionBuffer = [];
let currentHoldLetter = null;
let holdCounter = 0;
let lastConfirmedTime = 0;

// Stats
let frameCount = 0;
let lastFpsTime = Date.now();
let lastServerTime = 0;

// Connections
const CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [5, 9], [9, 10], [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [0, 17] // Palm base
];


// ---- Initialization ----

async function init() {
    try {
        // Setup webcam
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isRunning = true;
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected';
            requestAnimationFrame(processFrame);
        };
        
        // Setup TTS
        if ('speechSynthesis' in window) {
            console.log("Web Speech API supported.");
        } else {
            console.warn("Web Speech API not supported.");
            ttsEnabled = false;
            btnTtsToggle.innerHTML = "<span>&#128263;</span> TTS: Unsupported";
            btnTtsToggle.disabled = true;
        }

        // Setup Controls
        setupControls();

    } catch (err) {
        console.error("Camera error:", err);
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Camera Error: ' + err.message;
    }
}

function setupControls() {
    btnSpeak.onclick = () => {
        if (currentWord) {
            speak(currentWord);
            historyContent += currentWord + " ";
            historyText.textContent = historyContent;
            currentWord = "";
            wordText.textContent = currentWord;
        }
    };
    
    btnClear.onclick = () => {
        currentWord = "";
        wordText.textContent = currentWord;
    };
    
    btnBackspace.onclick = () => {
        if (currentWord.length > 0) {
            currentWord = currentWord.slice(0, -1);
            wordText.textContent = currentWord;
        }
    };
    
    btnReset.onclick = () => {
        currentWord = "";
        historyContent = "";
        wordText.textContent = currentWord;
        historyText.textContent = historyContent;
        predictionBuffer = [];
        holdCounter = 0;
        updateHoldUI();
    };
    
    btnTtsToggle.onclick = () => {
        ttsEnabled = !ttsEnabled;
        if (ttsEnabled) {
            btnTtsToggle.innerHTML = "<span>&#128264;</span> TTS: ON";
            btnTtsToggle.classList.add("active");
        } else {
            btnTtsToggle.innerHTML = "<span>&#128263;</span> TTS: OFF";
            btnTtsToggle.classList.remove("active");
        }
    };
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        
        switch(e.key.toLowerCase()) {
            case ' ':
                e.preventDefault();
                btnSpeak.click();
                break;
            case 'backspace':
                e.preventDefault();
                btnBackspace.click();
                break;
            case 'c':
                btnClear.click();
                break;
            case 'r':
                btnReset.click();
                break;
            case 't':
                btnTtsToggle.click();
                break;
        }
    });
}


// ---- Main Loop ----

async function processFrame() {
    if (!isRunning) return;

    // Calculate FPS
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        fpsCounter.textContent = `${frameCount} fps`;
        frameCount = 0;
        lastFpsTime = now;
    }

    // Limit server requests to ~10 fps to prevent overload
    if (now - lastServerTime > 100) {
        lastServerTime = now;
        
        // Grab frame as base64
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = video.videoWidth;
        tmpCanvas.height = video.videoHeight;
        const tmpCtx = tmpCanvas.getContext('2d');
        tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);
        
        const base64Frame = tmpCanvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: base64Frame })
            });
            
            const data = await response.json();
            
            if (data.hand_detected) {
                noHandOverlay.classList.add('hidden');
                drawLandmarks(data.landmarks);
                handlePrediction(data.letter, data.confidence);
            } else {
                noHandOverlay.classList.remove('hidden');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                resetHold();
                updateLetterUI("?", 0);
            }
            
        } catch (err) {
            console.error("Prediction error:", err);
        }
    }

    requestAnimationFrame(processFrame);
}


// ---- Rendering ----

function drawLandmarks(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reverse X because of mirror video
    const w = canvas.width;
    const h = canvas.height;
    
    // Draw connections
    ctx.strokeStyle = "rgba(34, 197, 94, 0.6)";
    ctx.lineWidth = 3;
    
    for (const [i, j] of CONNECTIONS) {
        const pt1 = landmarks[i];
        const pt2 = landmarks[j];
        
        ctx.beginPath();
        ctx.moveTo(w - pt1[0] * w, pt1[1] * h);
        ctx.lineTo(w - pt2[0] * w, pt2[1] * h);
        ctx.stroke();
    }
    
    // Draw points
    ctx.fillStyle = "rgba(34, 197, 94, 1.0)";
    
    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        const cx = w - pt[0] * w;
        const cy = pt[1] * h;
        
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // Wrist is larger
        if (i === 0) {
            ctx.beginPath();
            ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
            ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
            ctx.fill();
            ctx.fillStyle = "rgba(34, 197, 94, 1.0)";
        }
    }
}

function updateLetterUI(letter, conf) {
    predLetter.textContent = letter;
    predConfidence.textContent = Math.round(conf * 100) + "%";
    
    bigLetter.textContent = letter;
    
    const pct = Math.round(conf * 100);
    confidenceBar.style.width = pct + "%";
    confidenceLabel.textContent = `Confidence: ${pct}%`;
    
    if (pct > 85) {
        predictionBadge.classList.add('high-conf');
        bigLetter.className = 'big-letter high';
    } else if (pct > 50) {
        predictionBadge.classList.remove('high-conf');
        bigLetter.className = 'big-letter medium';
    } else {
        predictionBadge.classList.remove('high-conf');
        bigLetter.className = 'big-letter low';
    }
}


// ---- Logic ----

function handlePrediction(rawLetter, confidence) {
    updateLetterUI(rawLetter, confidence);
    
    if (confidence < 0.85) {
        resetHold();
        return;
    }
    
    // Majority Vote Smoothing
    predictionBuffer.push(rawLetter);
    if (predictionBuffer.length > BUFFER_SIZE) {
        predictionBuffer.shift();
    }
    
    const counts = {};
    let maxCount = 0;
    let majorityLetter = null;
    
    for (const l of predictionBuffer) {
        counts[l] = (counts[l] || 0) + 1;
        if (counts[l] > maxCount) {
            maxCount = counts[l];
            majorityLetter = l;
        }
    }
    
    // Hold-to-confirm logic
    const now = Date.now();
    if (now - lastConfirmedTime < 1000) {
        // Prevent double entries immediately after confirm
        resetHold();
        return;
    }
    
    if (majorityLetter === currentHoldLetter) {
        holdCounter++;
    } else {
        currentHoldLetter = majorityLetter;
        holdCounter = 1;
    }
    
    updateHoldUI();
    
    if (holdCounter >= HOLD_FRAMES_REQUIRED) {
        confirmLetter(majorityLetter);
        resetHold();
        lastConfirmedTime = now;
        predictionBuffer = [];
    }
}

function updateHoldUI() {
    if (holdCounter > 0) {
        holdPanel.style.display = 'block';
        const pct = Math.min(100, Math.round((holdCounter / HOLD_FRAMES_REQUIRED) * 100));
        holdBar.style.width = pct + "%";
        holdLabel.textContent = `${pct}%`;
        
        if (pct === 100) {
            holdBar.className = 'hold-bar confirming';
        } else {
            holdBar.className = 'hold-bar';
        }
    } else {
        holdPanel.style.display = 'none';
        holdBar.style.width = "0%";
    }
}

function resetHold() {
    holdCounter = 0;
    currentHoldLetter = null;
    updateHoldUI();
}

function confirmLetter(letter) {
    if (letter === "space") {
        currentWord += " ";
    } else if (letter === "del") {
        if (currentWord.length > 0) {
            currentWord = currentWord.slice(0, -1);
        }
    } else {
        currentWord += letter;
    }
    
    wordText.textContent = currentWord;
    
    // Small visual feedback
    document.body.style.boxShadow = "inset 0 0 50px rgba(34,197,94,0.1)";
    setTimeout(() => document.body.style.boxShadow = "none", 200);
}

function speak(text) {
    if (!ttsEnabled || !('speechSynthesis' in window)) return;
    
    const msg = new SpeechSynthesisUtterance();
    msg.text = text;
    msg.rate = 1.0;
    msg.pitch = 1.0;
    window.speechSynthesis.speak(msg);
}

// Start
window.onload = init;
