/* ZSL Translator phone app: camera -> MediaPipe -> model -> text/speech.
 * Everything runs on the device; no network needed. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
  var stage = $("stage"), statusEl = $("status"), notice = $("notice");
  var signEl = $("sign"), barEl = $("bar"), hintEl = $("hint"), guessEl = $("guess"), textEl = $("text");

  var HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  var POSE_LINKS = [[11,12],[11,13],[13,15],[12,14],[14,16]];

  var facing = "environment", stream = null;
  var model = null, typist = null, meter = null, hands = null, pose = null;
  var lastHands = [], lastPose = null, frame = 0, lastT = null, busy = false;
  var autoSpeak = false;
  var stats = { frames: 0, start: 0 };
  window.__zsl = { stats: stats, get text() { return typist ? typist.text : ""; }, log: [] };

  function setStatus(text, cls) { statusEl.textContent = text; statusEl.className = "pill" + (cls ? " " + cls : ""); }
  function showNotice(text) { notice.textContent = text; notice.hidden = !text; }

  function speak(text) {
    text = (text || "").trim();
    if (!text) return;
    if (window.AndroidTTS && window.AndroidTTS.speak) { window.AndroidTTS.speak(text); return; }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }
  }

  function renderText() {
    var t = typist ? typist.text : "";
    if (!t) { textEl.innerHTML = '<span class="placeholder">Signs you hold still for half a second appear here.</span>'; return; }
    textEl.textContent = t;
    var c = document.createElement("span"); c.className = "caret"; c.innerHTML = "&nbsp;";
    textEl.appendChild(c);
    textEl.scrollTop = textEl.scrollHeight;
  }

  function renderGuess() {
    if (!typist || !typist.guess) {
      signEl.textContent = "–"; barEl.style.width = "0"; guessEl.classList.remove("sure");
      hintEl.textContent = lastHands.length ? "Hold still…" : "Show a sign";
      return;
    }
    if (typist.guess === ZSL.NOT_A_SIGN) {
      signEl.textContent = "–"; barEl.style.width = "0"; guessEl.classList.remove("sure");
      hintEl.textContent = "Not a sign";
      return;
    }
    signEl.textContent = typist.guess;
    barEl.style.width = Math.round(typist.conf * 100) + "%";
    var sure = typist.conf >= typist.minConf;
    guessEl.classList.toggle("sure", sure && typist.still);
    hintEl.textContent = !sure ? "Not sure yet" : typist.still ? Math.round(typist.conf * 100) + "% sure" : "Moving – hold still";
  }

  function draw() {
    var w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.lineCap = "round";
    if (lastPose) {
      ctx.strokeStyle = "rgba(201,222,207,0.7)"; ctx.lineWidth = Math.max(2, w / 300);
      POSE_LINKS.forEach(function (l) {
        var a = lastPose[l[0]], b = lastPose[l[1]];
        ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
      });
    }
    lastHands.forEach(function (hand, k) {
      ctx.strokeStyle = k === 0 ? "#F08A24" : "#6FB7FF"; ctx.lineWidth = Math.max(2, w / 220);
      HAND_LINKS.forEach(function (l) {
        ctx.beginPath(); ctx.moveTo(hand[l[0]].x * w, hand[l[0]].y * h); ctx.lineTo(hand[l[1]].x * w, hand[l[1]].y * h); ctx.stroke();
      });
      ctx.fillStyle = "#FFFFFF";
      hand.forEach(function (p) { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, Math.max(2, w / 260), 0, 6.283); ctx.fill(); });
    });
  }

  async function startCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }
      });
    } catch (e) {
      setStatus("No camera", "err");
      showNotice("The app needs the camera. Allow camera access in Settings > Apps > ZSL Translator > Permissions, then reopen it.");
      throw e;
    }
    video.srcObject = stream;
    await video.play();
    stage.classList.toggle("mirror", facing === "user");
    lastHands = []; lastPose = null;
  }

  async function step() {
    if (busy || video.readyState < 2 || !video.videoWidth) return;
    busy = true;
    try {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      }
      await hands.send({ image: video });
      if (frame % 2 === 0 || !lastPose) await pose.send({ image: video }); // shoulders move slowly
      frame++;
      var now = performance.now() / 1000;
      var dt = lastT == null ? 0 : Math.min(now - lastT, 0.5);
      lastT = now;
      var feats = ZSL.frameFeatures(lastHands, lastPose, video.videoWidth / video.videoHeight);
      var proba = feats ? model.predict(feats) : null;
      var speed = meter.update(lastHands, video.videoWidth / video.videoHeight, now);
      var ev = typist.update(proba, dt, speed);
      if (ev === "sign") { renderText(); window.__zsl.log.push(typist.text.trim().slice(-1) + "@" + (video.currentTime || now).toFixed(1)); }
      if (ev === "word") { renderText(); if (autoSpeak) speak(typist.text.trim().split(" ").pop()); }
      renderGuess();
      draw();
      stats.frames++;
    } catch (e) {
      console.error(e);
    } finally {
      busy = false;
    }
  }

  function loop() {
    step().finally(function () { requestAnimationFrame(loop); });
  }

  async function init() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Not supported", "err");
      showNotice("This browser can't use the camera.");
      return;
    }
    setStatus("Loading AI…");
    var res = await fetch("zsl_model.json");
    var json = await res.json();
    model = new ZSL.Model(json);
    typist = new ZSL.Typist(model.labels, { minConf: 0.6, hold: 0.5, gap: 1.0, still: 1.0 });
    meter = new ZSL.MotionMeter(0.2);

    hands = new Hands({ locateFile: function (f) { return "mediapipe/hands/" + f; } });
    hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5, selfieMode: false });
    hands.onResults(function (r) { lastHands = r.multiHandLandmarks || []; });
    await hands.initialize();  // one at a time: the two engines share loader names

    pose = new Pose({ locateFile: function (f) { return "mediapipe/pose/" + f; } });
    pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5, selfieMode: false });
    pose.onResults(function (r) { lastPose = r.poseLandmarks || null; });
    await pose.initialize();

    setStatus("Starting camera…");
    await startCamera();
    setStatus(model.labels.length + " signs ready", "ready");
    stats.start = performance.now();
    renderText();
    loop();
  }

  $("flip").addEventListener("click", function () {
    facing = facing === "environment" ? "user" : "environment";
    startCamera().catch(function () {});
  });
  $("speak").addEventListener("click", function () { speak(typist && typist.text); });
  $("space").addEventListener("click", function () { if (typist && typist.text && !/ $/.test(typist.text)) { typist.text += " "; renderText(); } });
  $("back").addEventListener("click", function () { if (typist) { typist.text = typist.text.slice(0, -1); typist.last = null; renderText(); } });
  $("clear").addEventListener("click", function () { if (typist) { typist.text = ""; typist.last = null; renderText(); } });
  $("auto").addEventListener("click", function () {
    autoSpeak = !autoSpeak;
    this.classList.toggle("on", autoSpeak);
    this.setAttribute("aria-pressed", String(autoSpeak));
  });

  init().catch(function (e) {
    console.error(e);
    if (statusEl.className.indexOf("err") < 0) { setStatus("Error", "err"); showNotice("Something went wrong while starting: " + (e && e.message ? e.message : e)); }
  });
})();
