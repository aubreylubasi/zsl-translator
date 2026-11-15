/* ZSL Translator core: features, model and typing logic.
 * Must match zsl/features.py and live.py exactly; the phone and the
 * desktop share one trained model (zsl_model.json, written by train.py). */
(function (root) {
  "use strict";
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

  /* Small neural network exported from scikit-learn by train.py */
  function Model(json) {
    if (json.feature_dim !== FEATURE_DIM) throw new Error("Model was made for a different app version");
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

  /* Turns a stream of per-frame guesses into typed text (same as live.py) */
  function Typist(labels, opts) {
    opts = opts || {};
    this.labels = labels;
    this.minConf = opts.minConf != null ? opts.minConf : 0.6;
    this.hold = opts.hold != null ? opts.hold : 0.5;
    this.gap = opts.gap != null ? opts.gap : 1.0;
    this.smooth = opts.smooth || 8;
    this.buf = [];
    this.text = ""; this.cand = null; this.candT = 0; this.last = null; this.noHand = 0;
    this.guess = null; this.conf = 0;
  }
  Typist.prototype.update = function (proba, dt) {
    if (!proba) {
      this.buf = []; this.cand = null; this.guess = null;
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
    this.guess = this.labels[top]; this.conf = avg[top];
    if (this.conf < this.minConf) { this.cand = null; return null; }
    if (top !== this.cand) {
      this.cand = top; this.candT = 0;
      if (top !== this.last) this.last = null;
    } else {
      this.candT += dt;
    }
    if (this.candT >= this.hold && top !== this.last) {
      var lab = this.labels[top];
      this.text += lab.length === 1 ? lab : lab + " ";
      this.last = top;
      return "sign";
    }
    return null;
  };

  var api = { frameFeatures: frameFeatures, Model: Model, Typist: Typist, FEATURE_DIM: FEATURE_DIM };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ZSL = api;
})(this);
