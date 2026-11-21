/* ZSL Translator core: features, model, motion and typing logic.
 * Must match zsl/features.py and zsl/typist.py exactly; the phone and the
 * desktop share one trained model (zsl_model.json, written by train.py). */
(function (root) {
  "use strict";
  var NOT_A_SIGN = "_";           // the model's "not a sign" label: never typed
  var MOVING_SIGNS = ["J", "Z"];  // letters signed with a movement get a looser limit
  var N_SLOTS = 2;
  var HAND_DIM = 21 * 3 + 2 + 1 + 1;
  var FEATURE_DIM = HAND_DIM * N_SLOTS;

  /* hands: array of hands, each 21 {x,y,z} in 0..1 image coordinates
   * pose:  33 {x,y,z} or null;  aspect: image width / height */
  function frameFeatures(hands, pose, aspect) {
    if (!hands || hands.length === 0) return null;
    var cx, cy, width;
    if (pose && pose[11] && pose[12]) {
      var lx = pose[11].x * aspect, ly = pose[11].y;
      var rx = pose[12].x * aspect, ry = pose[12].y;
      cx = (lx + rx) / 2; cy = (ly + ry) / 2;
      width = Math.max(Math.hypot(lx - rx, ly - ry), 1e-3);
    } else {
      cx = aspect / 2; cy = 0.6; width = 0.35;
    }
    // highest wrist first: the signing hand
    var sorted = hands.slice().sort(function (a, b) { return a[0].y - b[0].y; }).slice(0, N_SLOTS);
    var out = new Float32Array(FEATURE_DIM);
    for (var s = 0; s < sorted.length; s++) {
      var h = sorted[s], o = s * HAND_DIM;
      var wx = h[0].x * aspect, wy = h[0].y, wz = h[0].z * aspect;
      var palm = Math.max(Math.hypot(h[9].x * aspect - wx, h[9].y - wy), 1e-4);
      for (var i = 0; i < 21; i++) {
        out[o + 3 * i] = (h[i].x * aspect - wx) / palm;
        out[o + 3 * i + 1] = (h[i].y - wy) / palm;
        out[o + 3 * i + 2] = (h[i].z * aspect - wz) / palm;
      }
      out[o + 63] = (wx - cx) / width;
      out[o + 64] = (wy - cy) / width;
      out[o + 65] = palm / width;
      out[o + 66] = 1;
    }
    return out;
  }

  /* Can we trust the body position? The model measures the hand against the
   * shoulders, so without real shoulders every guess is unreliable.
   * Returns null when the framing is fine, otherwise a short reason:
   *   "no-body"   no person found, or shoulders hidden or outside the picture
   *   "fake-body" the "shoulders" sit on the hand (the tracker mistook the hand for a body)
   *   "too-close" the hand is huge next to the shoulders (hand right up at the camera) */
  var MIN_VIS = 0.5, MAX_PALM_RATIO = 0.7;  // training data: 0.32 ± 0.06
  function framingIssue(hands, pose, aspect) {
    if (!hands || hands.length === 0) return null;
    if (!pose || !pose[11] || !pose[12]) return "no-body";
    var sh = [pose[11], pose[12]];
    for (var i = 0; i < 2; i++) {
      var p = sh[i];
      if (p.visibility != null && p.visibility < MIN_VIS) return "no-body";
      if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return "no-body";
    }
    var width = Math.max(Math.hypot((sh[0].x - sh[1].x) * aspect, sh[0].y - sh[1].y), 1e-3);
    for (var k = 0; k < hands.length; k++) {
      var h = hands[k], x0 = 1, x1 = 0, y0 = 1, y1 = 0;
      for (var j = 0; j < 21; j++) {
        x0 = Math.min(x0, h[j].x); x1 = Math.max(x1, h[j].x);
        y0 = Math.min(y0, h[j].y); y1 = Math.max(y1, h[j].y);
      }
      for (i = 0; i < 2; i++) {
        if (sh[i].x > x0 && sh[i].x < x1 && sh[i].y > y0 && sh[i].y < y1) return "fake-body";
      }
      var palm = Math.hypot((h[9].x - h[0].x) * aspect, h[9].y - h[0].y);
      if (palm / width > MAX_PALM_RATIO) return "too-close";
    }
    return null;
  }

  /* Motion window: the last `seconds` of signing as `frames` snapshots side by side,
   * oldest first. Same maths as Window in zsl/features.py. */
  var WINDOW_STALE = 0.25;
  function Window(frames, seconds) { this.k = frames; this.w = seconds; this.hist = []; }
  Window.prototype.reset = function () { this.hist = []; };
  Window.prototype.update = function (t, feats) {
    if (!feats) return null;
    var cut = t - this.w - WINDOW_STALE;
    this.hist = this.hist.filter(function (e) { return e[0] >= cut; });
    this.hist.push([t, feats]);
    var n = feats.length, out = new Float32Array(n * this.k), j = 0;
    for (var i = 0; i < this.k; i++) {
      var ts = t - this.w * (this.k - 1 - i) / (this.k - 1);
      while (j + 1 < this.hist.length && this.hist[j + 1][0] <= ts + 1e-9) j++;
      out.set(this.hist[j][1], i * n);
    }
    return out;
  };

  /* Small neural network exported from scikit-learn by train.py */
  function Model(json) {
    this.moving = json.moving || [];
    this.window = json.window || null;  // {frames, seconds} for a motion model
    if (json.feature_dim !== FEATURE_DIM * (this.window ? this.window.frames : 1)) throw new Error("Model was made for a different app version");
    this.labels = json.labels;
    this.mean = Float64Array.from(json.mean);
    this.scale = Float64Array.from(json.scale);
    this.layers = json.layers.map(function (L) {
      return { n: L["in"], m: L.out, W: Float64Array.from(L.W), b: Float64Array.from(L.b) };
    });
  }
  Model.prototype.predict = function (x) {
    var a = new Float64Array(x.length), i, j;
    for (i = 0; i < x.length; i++) a[i] = (x[i] - this.mean[i]) / this.scale[i];
    for (var k = 0; k < this.layers.length; k++) {
      var L = this.layers[k], z = Float64Array.from(L.b);
      for (i = 0; i < L.n; i++) {
        var ai = a[i];
        if (ai === 0) continue;
        var row = i * L.m;
        for (j = 0; j < L.m; j++) z[j] += ai * L.W[row + j];
      }
      if (k < this.layers.length - 1) {
        for (j = 0; j < L.m; j++) if (z[j] < 0) z[j] = 0;
      } else {
        var mx = -Infinity, sum = 0;
        for (j = 0; j < L.m; j++) if (z[j] > mx) mx = z[j];
        for (j = 0; j < L.m; j++) { z[j] = Math.exp(z[j] - mx); sum += z[j]; }
        for (j = 0; j < L.m; j++) z[j] /= sum;
      }
      a = z;
    }
    return a;
  };

  /* How fast the signing hand (the highest one) moves, in palm-lengths per second.
   * Compares the hand now with where it was `window` seconds ago, so tracker
   * jitter doesn't count as movement (same as zsl/typist.py MotionMeter). */
  function MotionMeter(window) {
    this.window = window != null ? window : 0.2;
    this.hist = [];
    this.speed = null;
  }
  MotionMeter.prototype.reset = function () { this.hist = []; this.speed = null; };
  MotionMeter.prototype.update = function (hands, aspect, t) {
    if (!hands || hands.length === 0) { this.reset(); return null; }
    var hand = hands[0], k, i;
    for (k = 1; k < hands.length; k++) if (hands[k][0].y < hand[0].y) hand = hands[k];
    var pts = new Float64Array(42);
    for (i = 0; i < 21; i++) { pts[2 * i] = hand[i].x * aspect; pts[2 * i + 1] = hand[i].y; }
    var palm = Math.max(Math.hypot(pts[18] - pts[0], pts[19] - pts[1]), 1e-4);
    this.hist.push({ t: t, pts: pts });
    while (this.hist.length > 2 && t - this.hist[1].t >= this.window) this.hist.shift();
    var ref = this.hist[0];
    if (t - ref.t < this.window / 2) {
      this.speed = null;
    } else {
      var moved = 0;
      for (i = 0; i < 21; i++) moved += Math.hypot(pts[2 * i] - ref.pts[2 * i], pts[2 * i + 1] - ref.pts[2 * i + 1]);
      this.speed = moved / 21 / palm / (t - ref.t);
    }
    return this.speed;
  };

  /* Turns a stream of per-frame guesses into typed text (same as zsl/typist.py).
   * Types when the same guess lasted `hold` s and the hand was still for the last `stillTime` s. */
  function Typist(labels, opts) {
    opts = opts || {};
    this.labels = labels;
    this.minConf = opts.minConf != null ? opts.minConf : 0.6;
    this.hold = opts.hold != null ? opts.hold : 0.5;
    this.gap = opts.gap != null ? opts.gap : 1.0;
    this.smooth = opts.smooth || 8;
    this.stillLimit = opts.still != null ? opts.still : 1.0;
    this.stillTime = opts.stillTime != null ? opts.stillTime : 0.15;
    this.movingSigns = opts.movingSigns || MOVING_SIGNS;
    this.movingFactor = opts.movingFactor != null ? opts.movingFactor : 3.0;
    // word signs with a movement (from the model's "moving" list): no still rule
    this.freeMove = (opts.freeMove || []).filter(function (l) { return l !== "J" && l !== "Z"; });
    this.moveHold = opts.moveHold != null ? opts.moveHold : 0.3;
    this.buf = [];
    this.text = ""; this.cand = null; this.candT = 0; this.last = null; this.noHand = 0;
    this.guess = null; this.conf = 0; this.still = false; this.stillT = 0;
  }
  /* Forget the current guess without ending the word (used while the framing is bad). */
  Typist.prototype.pause = function () {
    this.buf = []; this.cand = null; this.guess = null; this.conf = 0; this.still = false; this.stillT = 0;
  };
  /* speed: MotionMeter speed, or null if unknown */
  Typist.prototype.update = function (proba, dt, speed) {
    if (!proba) {
      this.buf = []; this.cand = null; this.guess = null; this.still = false;
      this.noHand += dt;
      if (this.noHand >= this.gap) {
        this.last = null;
        if (this.text && this.text[this.text.length - 1] !== " ") { this.text += " "; return "word"; }
      }
      return null;
    }
    this.noHand = 0;
    this.buf.push(proba);
    if (this.buf.length > this.smooth) this.buf.shift();
    var n = proba.length, avg = new Float64Array(n), top = 0, i;
    for (var b = 0; b < this.buf.length; b++) for (i = 0; i < n; i++) avg[i] += this.buf[b][i] / this.buf.length;
    for (i = 1; i < n; i++) if (avg[i] > avg[top]) top = i;
    var lab = this.labels[top];
    this.guess = lab; this.conf = avg[top];
    var limit = this.stillLimit * (this.movingSigns.indexOf(lab) >= 0 ? this.movingFactor : 1);
    this.still = this.stillLimit <= 0 || this.freeMove.indexOf(lab) >= 0 || (speed != null && speed <= limit);
    if (lab === NOT_A_SIGN || this.conf < this.minConf) { this.cand = null; return null; }
    if (top !== this.cand) {
      this.cand = top; this.candT = 0; this.stillT = 0;
      if (top !== this.last) this.last = null;
    } else {
      this.candT += dt;
      this.stillT = this.still ? this.stillT + dt : 0;
    }
    var settled = this.stillLimit <= 0 || this.stillT >= this.stillTime;
    var hold = this.freeMove.indexOf(lab) >= 0 ? this.moveHold : this.hold;
    if (this.candT >= hold && settled && top !== this.last) {
      this.text += lab.length === 1 ? lab : lab + " ";
      this.last = top;
      return "sign";
    }
    return null;
  };

  var api = { frameFeatures: frameFeatures, framingIssue: framingIssue, Window: Window, Model: Model, MotionMeter: MotionMeter, Typist: Typist,
    FEATURE_DIM: FEATURE_DIM, NOT_A_SIGN: NOT_A_SIGN };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ZSL = api;
})(this);
