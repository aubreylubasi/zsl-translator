#!/usr/bin/env bash
# Copies MediaPipe's web engines + models into www/ (so the app works offline)
# and puts www/ inside the Android project.
set -euo pipefail
cd "$(dirname "$0")"

npm install --no-audit --no-fund --silent

rm -rf www/mediapipe
mkdir -p www/mediapipe/hands www/mediapipe/pose
cp node_modules/@mediapipe/hands/hands.js \
   node_modules/@mediapipe/hands/hands.binarypb \
   node_modules/@mediapipe/hands/hand_landmark_full.tflite \
   node_modules/@mediapipe/hands/hands_solution_* \
   www/mediapipe/hands/
cp node_modules/@mediapipe/pose/pose.js \
   node_modules/@mediapipe/pose/pose_web.binarypb \
   node_modules/@mediapipe/pose/pose_landmark_full.tflite \
   node_modules/@mediapipe/pose/pose_solution_* \
   www/mediapipe/pose/

test -f www/zsl_model.json || { echo "www/zsl_model.json missing: run train.py first"; exit 1; }

rm -rf android/app/src/main/assets/www
mkdir -p android/app/src/main/assets
cp -R www android/app/src/main/assets/www
du -sh android/app/src/main/assets/www
