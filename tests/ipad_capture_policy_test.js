'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const policy = require('../relay/static/ipad_capture_policy.js');

test('host capture_config prefers the public top-level contract', () => {
  const config = policy.configFromStateMessage({
    capture_config: {fps: 12, resolution: {mode:'fixed', width:640, height:480}},
    capture: {capture_config: {target_fps:25, max_width:1280, max_height:720,
      adaptive_resolution:true}},
  }, policy.DEFAULT_CONFIG);
  assert.deepEqual(config, {
    target_fps:12, max_width:640, max_height:480, adaptive_resolution:false,
  });

  assert.deepEqual(policy.configFromStateMessage({
    capture_config: {fps:20, resolution:{mode:'auto'}},
  }, config), {
    target_fps:20, max_width:960, max_height:720, adaptive_resolution:true,
  });

  assert.deepEqual(policy.configFromStateMessage({
    capture: {capture_config: {fps:18,
      resolution:{mode:'fixed', width:800, height:600}}},
  }, policy.DEFAULT_CONFIG), {
    target_fps:18, max_width:800, max_height:600, adaptive_resolution:false,
  });
});

test('automatic profiles preserve aspect, orientation, even dimensions, and never upscale', () => {
  assert.deepEqual(policy.buildProfiles(policy.DEFAULT_CONFIG, 960, 720), [
    {width:960, height:720},
    {width:800, height:600},
    {width:640, height:480},
  ]);
  assert.deepEqual(policy.buildProfiles(policy.DEFAULT_CONFIG, 720, 960), [
    {width:720, height:960},
    {width:600, height:800},
    {width:480, height:640},
  ]);

  const wide = policy.buildProfiles(policy.DEFAULT_CONFIG, 1920, 1080);
  assert.deepEqual(wide, [
    {width:960, height:540},
    {width:800, height:450},
    {width:640, height:360},
  ]);
  wide.forEach(({width, height}) => {
    assert.equal(width % 2, 0);
    assert.equal(height % 2, 0);
  });
  assert.deepEqual(policy.buildProfiles(policy.DEFAULT_CONFIG, 320, 240), [
    {width:320, height:240},
  ]);
});

test('adaptation needs two pressured low windows and ignores throttling without pressure', () => {
  const adaptive = policy.createAdaptivePolicy(policy.DEFAULT_CONFIG);
  adaptive.setSource(960, 720);
  const pressuredLow = {sent_fps:10, encode_p90_ms:30,
    encode_busy_skips:1, backpressure_skips:0};
  assert.equal(adaptive.observeWindow(pressuredLow, false).changed, false);
  assert.equal(adaptive.observeWindow(pressuredLow, false).direction, 'down');
  assert.deepEqual(adaptive.getSize(), {width:800, height:600});

  const unpressuredLow = {sent_fps:2, encode_p90_ms:2,
    encode_busy_skips:0, backpressure_skips:0};
  for(let i = 0; i < 8; i++) adaptive.observeWindow(unpressuredLow, false);
  assert.deepEqual(adaptive.getSize(), {width:800, height:600});

  for(let i = 0; i < 4; i++) adaptive.observeWindow(pressuredLow, true);
  assert.deepEqual(adaptive.getSize(), {width:800, height:600});
});

test('deadline pacing selects about 20fps from a 24fps camera without aliasing to 12fps', () => {
  let deadline = 0;
  let sent = 0;
  const cameraInterval = 1000 / 24;
  const targetInterval = 1000 / 20;
  for(let callback = 0; callback < 240; callback++){
    const now = callback * cameraInterval;
    if(!deadline || now >= deadline){
      sent++;
      deadline = policy.nextFrameDeadline(deadline, now, targetInterval);
    }
  }
  assert.ok(sent >= 198 && sent <= 201, `expected about 200 frames, got ${sent}`);
});

test('signalling loss preserves an already healthy peer media path', () => {
  assert.equal(policy.mediaPathHealthy('open', 'connected', 'connected'), true);
  assert.equal(policy.mediaPathHealthy('closed', 'connected', 'connected'), false);
  assert.equal(policy.mediaPathHealthy('open', 'failed', 'connected'), false);
  assert.equal(policy.mediaPathHealthy('open', 'connected', 'failed'), false);
});

test('frame flow admits exactly one JPEG until its matching receiver ACK', () => {
  const flow = policy.createFrameFlowController(1000);
  assert.equal(flow.canAdmit(), true);
  assert.equal(flow.markSent(17, 100), true);
  assert.equal(flow.canAdmit(), false);
  assert.equal(flow.markSent(18, 110), false);
  assert.equal(flow.acknowledge(18, 120), null);
  assert.equal(flow.acknowledge(undefined, 120), null);
  assert.equal(flow.pendingAgeMs(1099), 999);
  assert.equal(flow.isTimedOut(1099), false);
  assert.equal(flow.isTimedOut(1100), true);
  assert.equal(flow.acknowledge(17, 1125), 1025);
  assert.equal(flow.canAdmit(), true);
  flow.reset();
  assert.equal(flow.pendingAgeMs(2000), 0);
});

test('frame flow with a window keeps several JPEGs in flight and tracks the oldest', () => {
  const flow = policy.createFrameFlowController(1000, 3);
  // Three frames may be admitted back-to-back before any ACK returns.
  assert.equal(flow.markSent(1, 100), true);
  assert.equal(flow.markSent(2, 110), true);
  assert.equal(flow.canAdmit(), true);
  assert.equal(flow.markSent(3, 120), true);
  assert.equal(flow.canAdmit(), false);        // window full at 3
  assert.equal(flow.markSent(4, 130), false);
  // Timeout is measured against the OLDEST unacknowledged frame (seq 1 @100).
  assert.equal(flow.pendingAgeMs(1099), 999);
  assert.equal(flow.isTimedOut(1100), true);
  // Out-of-order ACK frees exactly one slot and returns that frame's latency.
  assert.equal(flow.acknowledge(2, 210), 100);
  assert.equal(flow.canAdmit(), true);
  assert.equal(flow.acknowledge(2, 220), null);  // no double-ack
  // With seq 1 still pending, it remains the oldest for the timeout clock.
  assert.equal(flow.pendingAgeMs(600), 500);
  assert.equal(flow.acknowledge(1, 300), 200);
  // seq 3 is now the oldest.
  assert.equal(flow.pendingAgeMs(320), 200);
  assert.equal(flow.markSent(5, 330), true);     // room again after two acks
});

test('adaptation steps up after five healthy latency-adequate windows', () => {
  const adaptive = policy.createAdaptivePolicy(policy.DEFAULT_CONFIG);
  adaptive.setSource(960, 720);
  const pressuredLow = {sent_fps:10, encode_p90_ms:30,
    encode_busy_skips:0, backpressure_skips:1};
  adaptive.observeWindow(pressuredLow, false);
  adaptive.observeWindow(pressuredLow, false);
  assert.equal(adaptive.getTier(), 1);

  const slowHealthy = {sent_fps:20, encode_p90_ms:26, ack_p90_ms:60,
    encode_busy_skips:0, backpressure_skips:0, transport_wait_skips:40};
  for(let i = 0; i < 6; i++) adaptive.observeWindow(slowHealthy, false);
  assert.equal(adaptive.getTier(), 1);

  const healthy = Object.assign({}, slowHealthy, {encode_p90_ms:20, ack_p90_ms:20});
  for(let i = 0; i < 4; i++) assert.equal(adaptive.observeWindow(healthy, false).changed, false);
  assert.equal(adaptive.observeWindow(healthy, false).direction, 'up');
  assert.equal(adaptive.getTier(), 0);
});

test('fixed mode stays q0.92 while auto preserves 640 pixels before reducing JPEG quality', () => {
  const low = {sent_fps:5, encode_p90_ms:80,
    encode_busy_skips:1, backpressure_skips:1};
  const fixed = policy.createAdaptivePolicy({target_fps:20, max_width:640,
    max_height:480, adaptive_resolution:false});
  fixed.setSource(960, 720);
  for(let i = 0; i < 10; i++) fixed.observeWindow(low, false);
  assert.deepEqual(fixed.getSize(), {width:640, height:480});
  assert.equal(fixed.getJpegQuality(), 0.92);
  assert.equal(fixed.isConstrained(), false);

  const automatic = policy.createAdaptivePolicy(policy.DEFAULT_CONFIG);
  automatic.setSource(960, 720);
  for(let i = 0; i < 6; i++) automatic.observeWindow(low, false);
  assert.deepEqual(automatic.getSize(), {width:640, height:480});
  assert.equal(automatic.getJpegQuality(), 0.88);
  assert.equal(automatic.isConstrained(), false);
  for(let i = 0; i < 10; i++) automatic.observeWindow(low, false);
  assert.equal(automatic.getJpegQuality(), 0.72);
  assert.equal(automatic.isConstrained(), true);
});

test('healthy ACK cadence restores JPEG quality before spatial resolution', () => {
  const adaptive = policy.createAdaptivePolicy(policy.DEFAULT_CONFIG);
  adaptive.setSource(960, 720);
  const low = {sent_fps:5, encode_p90_ms:10, ack_p90_ms:300,
    encode_busy_skips:0, backpressure_skips:0, transport_wait_skips:50};
  for(let i = 0; i < 6; i++) adaptive.observeWindow(low, false);
  assert.deepEqual(adaptive.getSize(), {width:640, height:480});
  assert.equal(adaptive.getJpegQuality(), 0.88);

  const healthy = {sent_fps:20, encode_p90_ms:15, ack_p90_ms:25,
    encode_busy_skips:0, backpressure_skips:0, transport_wait_skips:20};
  for(let i = 0; i < 5; i++) adaptive.observeWindow(healthy, false);
  assert.equal(adaptive.getJpegQuality(), 0.92);
  assert.deepEqual(adaptive.getSize(), {width:640, height:480});
  for(let i = 0; i < 5; i++) adaptive.observeWindow(healthy, false);
  assert.deepEqual(adaptive.getSize(), {width:800, height:600});
});

test('sender stats have an exact bounded scalar schema', () => {
  assert.deepEqual(policy.boundedSenderStats({
    sent_fps:Infinity, target_fps:20.123, width:99999, height:-1, tier:99,
    quality_tier:99, jpeg_quality:Infinity,
    encode_p90_ms:70000, jpeg_bytes:Infinity, buffered_bytes:40000000,
    encode_busy_skips:100001, backpressure_skips:-2, transport_wait_skips:100002,
    ack_p90_ms:70001, ack_timeouts:-1, constrained:1,
  }), {
    type:'sender_stats', sent_fps:0, target_fps:20.12, width:4096, height:2,
    tier:16, quality_tier:16, jpeg_quality:0.92, encode_p90_ms:60000,
    jpeg_bytes:0, buffered_bytes:33554432,
    encode_busy_skips:100000, backpressure_skips:0, transport_wait_skips:100000,
    ack_p90_ms:60000, ack_timeouts:0, constrained:false,
  });
});

test('capture profile boundary has an exact bounded scalar schema', () => {
  assert.deepEqual(policy.boundedCaptureProfile({
    width:99999, height:-1, tier:99, quality_tier:-3, jpeg_quality:Infinity,
    raw_pixels:'must not cross the boundary',
  }), {
    type:'capture_profile', width:4096, height:2, tier:16,
    quality_tier:0, jpeg_quality:0.92,
  });
});

test('deployed page keeps the required transport and single-flight contracts', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'relay', 'static', 'ipad.html'), 'utf8');
  assert.doesNotMatch(html, /\.addTrack\s*\(/);
  assert.match(html, /createDataChannel\('frames', \{ordered:true\}\)/);
  assert.match(html, /const BUFFERED_AMOUNT_LIMIT = 0/);
  assert.match(html, /const MAX_MESSAGE_BYTES = 60 \* 1024/);
  assert.match(html, /const MAX_CHUNK_BYTES = MAX_MESSAGE_BYTES - FRAME_HEADER_BYTES/);
  assert.match(html, /CapturePolicy\.createFrameFlowController\(FRAME_ACK_TIMEOUT_MS, FRAME_WINDOW\)/);
  assert.match(html, /const FRAME_ACK_TIMEOUT_MS = 2500/);
  assert.match(html, /const FRAME_WINDOW = \d+/);
  assert.match(html, /scheduleReconnect\('frame ack timeout'\)/);
  assert.match(html, /message\.type !== 'frame_ack'/);
  assert.match(html, /capturePolicy\.getJpegQuality\(\)/);
  assert.match(html, /scheduleReconnect\('frames closed'\)/);
  assert.match(html, /CapturePolicy\.mediaPathHealthy/);
  assert.match(html, /frameChannel\.send\(JSON\.stringify\(profile\)\)/);
  assert.match(html, /publishCaptureProfileBoundary\(true\)/);
  assert.match(html, /encodeInFlight \|\| !frameFlow\.canAdmit\(\)/);
  assert.match(html, /if\(!publishCaptureProfileBoundary\(false\)\) return;/);
  assert.match(html, /scheduleReconnect\('capture profile boundary failed'\)/);
  assert.match(html, /let captureGeneration = 0/);
  assert.match(html, /let encodeInFlight = false/);
  const establish = html.match(
    /async function establishPeer\(\)\{([\s\S]*?)\n  \}\n\n  async function connect/);
  assert.ok(establish, 'establishPeer function must remain inspectable');
  assert.match(establish[1], /try\{/);
  assert.match(establish[1], /catch\(err\)\{[\s\S]*?establishing = false;[\s\S]*?throw err;/);
  assert.equal((html.match(/canvas\.toBlob\s*\(/g) || []).length, 1);
  const inlineScripts = Array.from(html.matchAll(/<script>([\s\S]*?)<\/script>/g));
  assert.equal(inlineScripts.length, 1);
  assert.doesNotThrow(() => new Function(inlineScripts[0][1]));
});
