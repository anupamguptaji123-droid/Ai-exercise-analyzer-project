// Global variables
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let mediaStream = null;
let isRunning = false;
let repCount = 0;
let previousFormQuality = null;
let exerciseType = 'squat';

const angleElements = {
    leftKnee: document.getElementById('leftKnee'),
    rightKnee: document.getElementById('rightKnee'),
    leftElbow: document.getElementById('leftElbow'),
    rightElbow: document.getElementById('rightElbow'),
    spineAngle: document.getElementById('spineAngle'),
    hipAngle: document.getElementById('hipAngle')
};

// Start camera
function startCamera() {
    navigator.mediaDevices
        .getUserMedia({ video: { width: 640, height: 480 } })
        .then(stream => {
            mediaStream = stream;
            video.srcObject = stream;
            isRunning = true;
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                startPoseDetection();
            };
        })
        .catch(err => {
            alert('Camera access denied: ' + err.message);
            console.log(err);
        });
}

// Stop camera
function stopCamera() {
    isRunning = false;
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    updateFeedback('Camera stopped', 'warning');
}

// Reset statistics
function resetStats() {
    repCount = 0;
    document.getElementById('repCount').textContent = '0';
    updateFeedback('Statistics reset', 'good');
}

// Update exercise selection
document.getElementById('exerciseSelect').addEventListener('change', (e) => {
    exerciseType = e.target.value;
    resetStats();
});

// Start pose detection
async function startPoseDetection() {
    // Load TensorFlow.js and PoseNet
    const script1 = document.createElement('script');
    script1.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs';
    script1.onload = () => {
        const script2 = document.createElement('script');
        script2.src = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet';
        script2.onload = () => {
            detectPose();
        };
        document.body.appendChild(script2);
    };
    document.body.appendChild(script1);
}

// Pose detection loop
async function detectPose() {
    if (!isRunning) return;

    try {
        const net = await posenet.load();
        
        const detectFrame = async () => {
            if (!isRunning) return;
            
            const pose = await net.estimateSinglePose(video, {
                flipHorizontal: true
            });
            
            // Clear canvas and draw
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw pose
            drawPose(pose);
            
            // Analyze pose
            analyzePose(pose);
            
            requestAnimationFrame(detectFrame);
        };
        
        detectFrame();
    } catch (error) {
        console.error('Pose detection error:', error);
        setTimeout(detectPose, 1000);
    }
}

// Draw pose landmarks
function drawPose(pose) {
    const keypoints = pose.keypoints;
    
    // Color gradient for joints
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#AED6F1'
    ];
    
    // Draw keypoints
    keypoints.forEach((keypoint, idx) => {
        if (keypoint.score > 0.5) {
            const { position } = keypoint;
            
            // Draw circle
            ctx.fillStyle = colors[idx % colors.length];
            ctx.beginPath();
            ctx.arc(position.x, position.y, 5, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw label
            ctx.fillStyle = '#FFF';
            ctx.font = 'bold 12px Arial';
            ctx.fillText(getKeypointName(idx), position.x + 8, position.y);
        }
    });
    
    // Draw skeleton lines
    const connections = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12], [11, 13], [13, 15],
        [12, 14], [14, 16]
    ];
    
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    connections.forEach(([start, end]) => {
        const startPoint = keypoints[start].position;
        const endPoint = keypoints[end].position;
        if (keypoints[start].score > 0.5 && keypoints[end].score > 0.5) {
            ctx.beginPath();
            ctx.moveTo(startPoint.x, startPoint.y);
            ctx.lineTo(endPoint.x, endPoint.y);
            ctx.stroke();
        }
    });
}

// Get keypoint names
function getKeypointName(idx) {
    const names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];
    return names[idx] || '';
}

// Calculate angle between three points
function calculateAngle(pointA, pointB, pointC) {
    const radians = Math.atan2(pointC.y - pointB.y, pointC.x - pointB.x) -
                   Math.atan2(pointA.y - pointB.y, pointA.x - pointB.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return Math.round(angle);
}

// Analyze pose for exercise
function analyzePose(pose) {
    const keypoints = pose.keypoints;
    
    // Get key body parts
    const leftShoulder = keypoints[5];
    const rightShoulder = keypoints[6];
    const leftElbow = keypoints[7];
    const rightElbow = keypoints[8];
    const leftHip = keypoints[11];
    const rightHip = keypoints[12];
    const leftKnee = keypoints[13];
    const rightKnee = keypoints[14];
    const leftAnkle = keypoints[15];
    const rightAnkle = keypoints[16];
    
    // Calculate angles
    const leftKneeAngle = calculateAngle(
        leftHip.position, leftKnee.position, leftAnkle.position
    );
    const rightKneeAngle = calculateAngle(
        rightHip.position, rightKnee.position, rightAnkle.position
    );
    const leftElbowAngle = calculateAngle(
        leftShoulder.position, leftElbow.position, {x: leftElbow.position.x + 100, y: leftElbow.position.y}
    );
    const rightElbowAngle = calculateAngle(
        rightShoulder.position, rightElbow.position, {x: rightElbow.position.x + 100, y: rightElbow.position.y}
    );
    
    // Update angle display
    angleElements.leftKnee.textContent = leftKneeAngle + '°';
    angleElements.rightKnee.textContent = rightKneeAngle + '°';
    angleElements.leftElbow.textContent = leftElbowAngle + '°';
    angleElements.rightElbow.textContent = rightElbowAngle + '°';
    
    // Analyze based on exercise type
    analyzeExerciseForm(exerciseType, leftKneeAngle, rightKneeAngle, leftElbowAngle, rightElbowAngle);
}

// Analyze exercise form
function analyzeExerciseForm(exercise, leftKnee, rightKnee, leftElbow, rightElbow) {
    let feedback = '';
    let quality = 'bad';
    
    switch(exercise) {
        case 'squat':
            if (leftKnee < 90 && rightKnee < 90) {
                feedback = '✅ Great squat depth! Keep back straight.';
                quality = 'good';
            } else if (leftKnee < 110 && rightKnee < 110) {
                feedback = '⚠️ Go lower for better squat depth!';
                quality = 'warning';
            } else {
                feedback = '❌ Squat deeper - knees should bend more!';
                quality = 'bad';
            }
            break;
            
        case 'pushup':
            if (leftElbow > 70 && rightElbow > 70) {
                feedback = '✅ Perfect push-up form!';
                quality = 'good';
            } else if (leftElbow > 60 && rightElbow > 60) {
                feedback = '⚠️ Lower your body more for full range!';
                quality = 'warning';
            } else {
                feedback = '❌ Elbows too bent - proper form needed!';
                quality = 'bad';
            }
            break;
            
        case 'deadlift':
            feedback = '⚠️ Keep back straight and lift with legs!';
            quality = 'warning';
            break;
            
        case 'shoulder-press':
            if (leftElbow > 160 && rightElbow > 160) {
                feedback = '✅ Excellent shoulder press form!';
                quality = 'good';
            } else {
                feedback = '⚠️ Press weights higher for full extension!';
                quality = 'warning';
            }
            break;
    }
    
    updateFeedback(feedback, quality);
    
    // Update form quality
    const qualityText = quality === 'good' ? '✅ Good' : quality === 'warning' ? '⚠️ Fair' : '❌ Needs Work';
    document.getElementById('formQuality').textContent = qualityText;
    
    // Count reps (simple detection when form improves)
    if (quality === 'good' && previousFormQuality !== 'good') {
        repCount++;
        document.getElementById('repCount').textContent = repCount;
    }
    previousFormQuality = quality;
}

// Update feedback display
function updateFeedback(message, type = 'warning') {
    const feedbackBox = document.getElementById('feedback');
    feedbackBox.innerHTML = `<p class="feedback-text">${message}</p>`;
    feedbackBox.className = 'feedback-box feedback-' + type;
    
    // Add stability indicator
    const stabilityText = type === 'good' ? '✅ Stable' : type === 'warning' ? '⚠️ Adjusting' : '❌ Unstable';
    document.getElementById('stability').textContent = stabilityText;
}