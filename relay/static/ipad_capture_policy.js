(function(root, factory){
  'use strict';
  if(typeof module === 'object' && module.exports){
    module.exports = factory();
  } else {
    root.IPadCapturePolicy = factory();
  }
}(typeof self !== 'undefined' ? self : this, function(){
  'use strict';

  const DEFAULT_CONFIG = Object.freeze({
    target_fps: 20,
    max_width: 960,
    max_height: 720,
    adaptive_resolution: true,
  });
  const AUTO_LONG_EDGES = Object.freeze([960, 800, 640]);
  const JPEG_QUALITIES = Object.freeze([0.92, 0.88, 0.84, 0.80, 0.76, 0.72]);
  const UPSHIFT_HEALTHY_WINDOWS = 15;
  const MIN_DIMENSION = 160;
  const MAX_DIMENSION = 4096;

  function finiteNumber(value, fallback, minimum, maximum){
    const parsed = Number(value);
    if(!Number.isFinite(parsed)) return fallback;
    return Math.min(maximum, Math.max(minimum, parsed));
  }

  function boundedInteger(value, fallback, minimum, maximum){
    return Math.round(finiteNumber(value, fallback, minimum, maximum));
  }

  function normalizeConfig(raw, fallback){
    const base = fallback || DEFAULT_CONFIG;
    const value = raw && typeof raw === 'object' ? raw : {};
    return {
      target_fps: finiteNumber(value.target_fps, base.target_fps, 1, 30),
      max_width: boundedInteger(value.max_width, base.max_width,
        MIN_DIMENSION, MAX_DIMENSION),
      max_height: boundedInteger(value.max_height, base.max_height,
        MIN_DIMENSION, MAX_DIMENSION),
      adaptive_resolution: typeof value.adaptive_resolution === 'boolean'
        ? value.adaptive_resolution : base.adaptive_resolution,
    };
  }

  function configFromStateMessage(message, fallback){
    if(!message || typeof message !== 'object') return null;
    const base = fallback || DEFAULT_CONFIG;

    function fromPublicShape(value){
      if(!value || typeof value !== 'object') return null;
      const resolution = value.resolution;
      if(!resolution || typeof resolution !== 'object') return null;
      if(resolution.mode === 'auto'){
        return normalizeConfig({
          target_fps: value.fps,
          max_width: DEFAULT_CONFIG.max_width,
          max_height: DEFAULT_CONFIG.max_height,
          adaptive_resolution: true,
        }, base);
      }
      if(resolution.mode === 'fixed'){
        return normalizeConfig({
          target_fps: value.fps,
          max_width: resolution.width,
          max_height: resolution.height,
          adaptive_resolution: false,
        }, base);
      }
      return null;
    }

    // Public host contract. Keep it top-level so the page need not understand
    // the camera diagnostics payload, which can evolve independently.
    const publicConfig = message.capture_config;
    if(publicConfig && typeof publicConfig === 'object'){
      const parsed = fromPublicShape(publicConfig);
      if(parsed) return parsed;
    }

    // Tolerate the earlier diagnostics-nested shape during rolling upgrades.
    const nested = message.capture && message.capture.capture_config;
    if(nested && typeof nested === 'object'){
      return fromPublicShape(nested) || normalizeConfig(nested, base);
    }
    return null;
  }

  function evenFloor(value){
    return Math.max(2, Math.floor(value / 2) * 2);
  }

  function fitEven(sourceWidth, sourceHeight, boxWidth, boxHeight){
    const sw = boundedInteger(sourceWidth, 2, 2, MAX_DIMENSION);
    const sh = boundedInteger(sourceHeight, 2, 2, MAX_DIMENSION);
    const bw = boundedInteger(boxWidth, sw, 2, MAX_DIMENSION);
    const bh = boundedInteger(boxHeight, sh, 2, MAX_DIMENSION);
    const scale = Math.min(1, bw / sw, bh / sh);
    return {width: evenFloor(sw * scale), height: evenFloor(sh * scale)};
  }

  function buildProfiles(configValue, sourceWidth, sourceHeight){
    const config = normalizeConfig(configValue, DEFAULT_CONFIG);
    const sw = boundedInteger(sourceWidth, config.max_width, 2, MAX_DIMENSION);
    const sh = boundedInteger(sourceHeight, config.max_height, 2, MAX_DIMENSION);
    let boxWidth = config.max_width;
    let boxHeight = config.max_height;
    if((sw < sh) !== (boxWidth < boxHeight)){
      const swap = boxWidth;
      boxWidth = boxHeight;
      boxHeight = swap;
    }

    const topLongEdge = Math.max(boxWidth, boxHeight);
    const longEdges = [topLongEdge];
    if(config.adaptive_resolution){
      AUTO_LONG_EDGES.forEach(function(edge){
        if(edge < topLongEdge && edge <= topLongEdge) longEdges.push(edge);
      });
    }

    const profiles = [];
    const seen = Object.create(null);
    longEdges.forEach(function(longEdge){
      const ratio = longEdge / topLongEdge;
      const fitted = fitEven(sw, sh, boxWidth * ratio, boxHeight * ratio);
      const key = fitted.width + 'x' + fitted.height;
      if(!seen[key]){
        seen[key] = true;
        profiles.push(fitted);
      }
    });
    return profiles.length ? profiles : [fitEven(sw, sh, boxWidth, boxHeight)];
  }

  function configsEqual(left, right){
    return left.target_fps === right.target_fps &&
      left.max_width === right.max_width &&
      left.max_height === right.max_height &&
      left.adaptive_resolution === right.adaptive_resolution;
  }

  function nextFrameDeadline(currentDeadline, now, frameInterval){
    const safeNow = finiteNumber(now, 0, 0, 1e12);
    const safeInterval = finiteNumber(frameInterval, 50, 1, 1000);
    const current = finiteNumber(currentDeadline, 0, 0, 1e12);
    return current && safeNow - current <= safeInterval
      ? current + safeInterval : safeNow + safeInterval;
  }

  function mediaPathHealthy(frameState, peerState, iceState){
    return frameState === 'open' && peerState !== 'failed' &&
      peerState !== 'closed' && iceState !== 'failed' && iceState !== 'closed';
  }

  // Bounded-window sender flow control over the receiver's per-frame ACKs.
  // A window of 1 is pure stop-and-wait; a larger window keeps several JPEGs
  // in flight so the next frame ships while an earlier ACK is still crossing
  // the wire. That hides the ACK round-trip (which on a busy host is the whole
  // throughput ceiling) while still bounding in-flight memory to window frames.
  function createFrameFlowController(timeoutMs, windowSize){
    const timeout = finiteNumber(timeoutMs, 1000, 100, 10000);
    const maxInFlight = boundedInteger(windowSize, 1, 1, 8);
    // Insertion-ordered seq -> sentAt. Map iteration order is insertion order,
    // so the first entry is always the oldest unacknowledged frame.
    const pending = new Map();

    function oldestSentAt(){
      const first = pending.values().next();
      return first.done ? 0 : first.value;
    }

    return {
      canAdmit: function(){ return pending.size < maxInFlight; },
      markSent: function(seq, now){
        if(pending.size >= maxInFlight) return false;
        const candidate = Number(seq);
        if(!Number.isInteger(candidate) || candidate < 0 || candidate > 0xFFFFFFFF) return false;
        const key = candidate >>> 0;
        if(pending.has(key)) return false;
        pending.set(key, finiteNumber(now, 0, 0, 1e12));
        return true;
      },
      acknowledge: function(seq, now){
        const candidate = Number(seq);
        if(!Number.isInteger(candidate) || candidate < 0 || candidate > 0xFFFFFFFF) return null;
        const key = candidate >>> 0;
        const sentAt = pending.get(key);
        if(sentAt === undefined) return null;
        pending.delete(key);
        return Math.max(0, finiteNumber(now, sentAt, 0, 1e12) - sentAt);
      },
      pendingAgeMs: function(now){
        if(pending.size === 0) return 0;
        return Math.max(0, finiteNumber(now, oldestSentAt(), 0, 1e12) - oldestSentAt());
      },
      isTimedOut: function(now){
        return pending.size > 0 && this.pendingAgeMs(now) >= timeout;
      },
      reset: function(){ pending.clear(); },
    };
  }

  function createAdaptivePolicy(initialConfig){
    let config = normalizeConfig(initialConfig, DEFAULT_CONFIG);
    let sourceWidth = config.max_width;
    let sourceHeight = config.max_height;
    let profiles = buildProfiles(config, sourceWidth, sourceHeight);
    let tier = 0;
    let qualityTier = 0;
    let lowPressureWindows = 0;
    let healthyWindows = 0;
    let constrained = false;

    function rebuild(resetTier){
      profiles = buildProfiles(config, sourceWidth, sourceHeight);
      tier = resetTier ? 0 : Math.min(tier, profiles.length - 1);
      qualityTier = resetTier ? 0 : Math.min(qualityTier, JPEG_QUALITIES.length - 1);
      lowPressureWindows = 0;
      healthyWindows = 0;
      constrained = false;
    }

    function change(direction){
      return {changed:direction !== null, direction:direction, tier:tier,
        qualityTier:qualityTier, constrained:constrained};
    }

    function downshift(allowFixedQuality){
      lowPressureWindows = 0;
      healthyWindows = 0;
      let direction = null;
      if(!config.adaptive_resolution){
        if(allowFixedQuality && qualityTier < JPEG_QUALITIES.length - 1){
          qualityTier += 1;
          constrained = false;
          direction = 'down';
        } else if(allowFixedQuality) {
          constrained = true;
        }
        return change(direction);
      }
      if(tier < profiles.length - 1){
        tier += 1;
        constrained = false;
        direction = 'down';
      } else if(qualityTier < JPEG_QUALITIES.length - 1){
        qualityTier += 1;
        constrained = false;
        direction = 'down';
      } else {
        constrained = true;
      }
      return change(direction);
    }

    return {
      getConfig: function(){ return Object.assign({}, config); },
      getProfiles: function(){ return profiles.map(function(p){ return Object.assign({}, p); }); },
      getTier: function(){ return tier; },
      getQualityTier: function(){ return qualityTier; },
      getJpegQuality: function(){ return JPEG_QUALITIES[qualityTier]; },
      getSize: function(){ return Object.assign({}, profiles[tier]); },
      isConstrained: function(){ return constrained; },
      forceDownshift: function(){ return downshift(true); },
      setConfig: function(value){
        const next = normalizeConfig(value, config);
        if(configsEqual(next, config)) return false;
        config = next;
        rebuild(true);
        return true;
      },
      setSource: function(width, height){
        const nextWidth = boundedInteger(width, sourceWidth, 2, MAX_DIMENSION);
        const nextHeight = boundedInteger(height, sourceHeight, 2, MAX_DIMENSION);
        if(nextWidth === sourceWidth && nextHeight === sourceHeight) return false;
        sourceWidth = nextWidth;
        sourceHeight = nextHeight;
        rebuild(false);
        return true;
      },
      observeWindow: function(sample, hidden){
        const data = sample && typeof sample === 'object' ? sample : {};
        const sentFps = finiteNumber(data.sent_fps, 0, 0, 60);
        const encodeP90 = finiteNumber(data.encode_p90_ms, 0, 0, 60000);
        const busy = boundedInteger(data.encode_busy_skips, 0, 0, 100000);
        const backpressure = boundedInteger(data.backpressure_skips, 0, 0, 100000);
        const transportWait = boundedInteger(data.transport_wait_skips, 0, 0, 100000);
        const ackTimeouts = boundedInteger(data.ack_timeouts, 0, 0, 100000);
        const ackP90 = finiteNumber(data.ack_p90_ms, 0, 0, 60000);
        const pressured = busy > 0 || backpressure > 0 || transportWait > 0 ||
          ackTimeouts > 0;
        const low = sentFps < config.target_fps * 0.85 && pressured;
        // Waiting briefly for the receiver acknowledgement is normal, bounded
        // flow control, not congestion. It may trigger a downshift when it
        // actually suppresses FPS, but must not prevent recovery to a higher
        // tier while acknowledgements and encoding both fit in one frame.
        const frameInterval = 1000 / config.target_fps;
        const healthy = sentFps >= config.target_fps * 0.95 && busy === 0 &&
          backpressure === 0 && encodeP90 <= frameInterval * 0.5 &&
          ackP90 <= frameInterval;
        let direction = null;

        if(hidden || !config.adaptive_resolution){
          lowPressureWindows = 0;
          healthyWindows = 0;
          return {changed:false, direction:null, tier:tier, qualityTier:qualityTier,
            constrained:constrained};
        }

        lowPressureWindows = low ? Math.min(2, lowPressureWindows + 1) : 0;
        healthyWindows = healthy
          ? Math.min(UPSHIFT_HEALTHY_WINDOWS, healthyWindows + 1) : 0;
        if(healthy) constrained = false;

        if(lowPressureWindows >= 2){
          return downshift(false);
        } else if(healthyWindows >= UPSHIFT_HEALTHY_WINDOWS &&
            (qualityTier > 0 || tier > 0)){
          if(qualityTier > 0) qualityTier -= 1;
          else tier -= 1;
          lowPressureWindows = 0;
          healthyWindows = 0;
          constrained = false;
          direction = 'up';
        }
        return change(direction);
      },
    };
  }

  // One unhealthy episode gets a cheap in-place recovery and at most one
  // peer rebuild.  It can be rearmed only by sustained healthy frame flow or
  // an explicit user retry; there is deliberately no recursive retry state.
  function createRecoveryController(healthyWindowTarget){
    const requiredHealthy = boundedInteger(healthyWindowTarget, 2, 1, 10);
    let ackTimeouts = 0;
    let reconnectUsed = false;
    let healthyWindows = 0;
    let manualRequired = false;

    function snapshot(action, rearmed){
      return {action:action || 'none', ackTimeouts:ackTimeouts,
        reconnectUsed:reconnectUsed, healthyWindows:healthyWindows,
        manualRequired:manualRequired, rearmed:rearmed === true};
    }

    function reset(){
      ackTimeouts = 0;
      reconnectUsed = false;
      healthyWindows = 0;
      manualRequired = false;
    }

    return {
      ackTimeout: function(){
        if(manualRequired) return snapshot('manual');
        healthyWindows = 0;
        ackTimeouts += 1;
        if(ackTimeouts === 1) return snapshot('recover');
        if(!reconnectUsed){
          reconnectUsed = true;
          return snapshot('reconnect');
        }
        manualRequired = true;
        return snapshot('manual');
      },
      terminalFailure: function(){
        if(manualRequired) return snapshot('manual');
        healthyWindows = 0;
        if(!reconnectUsed){
          reconnectUsed = true;
          return snapshot('reconnect');
        }
        manualRequired = true;
        return snapshot('manual');
      },
      reconnectFailed: function(){
        healthyWindows = 0;
        manualRequired = true;
        return snapshot('manual');
      },
      healthyWindow: function(healthy){
        if(manualRequired) return snapshot('manual');
        if(!healthy){
          healthyWindows = 0;
          return snapshot('none');
        }
        if(ackTimeouts === 0 && !reconnectUsed) return snapshot('none');
        healthyWindows += 1;
        if(healthyWindows < requiredHealthy) return snapshot('none');
        reset();
        return snapshot('none', true);
      },
      manualRetry: function(){
        reset();
        return snapshot('none');
      },
      snapshot: function(){ return snapshot('none'); },
    };
  }

  // HR and SpO2 may disappear from telemetry while their six-second signal
  // windows refill.  Retain selected last-known values only in page memory so
  // the UI can label them stale without changing freshness or wire contracts.
  function createHeldValueCache(ids){
    const allowed = Object.create(null);
    (Array.isArray(ids) ? ids : []).forEach(function(id){ allowed[String(id)] = true; });
    const held = Object.create(null);
    return {
      merge: function(values){
        const seen = Object.create(null);
        const merged = (Array.isArray(values) ? values : []).map(function(raw){
          const value = raw && typeof raw === 'object' ? Object.assign({}, raw) : {};
          const id = String(value.id || '');
          seen[id] = true;
          if(!allowed[id]) return value;
          if(value.present && value.value !== null && value.value !== undefined){
            held[id] = {value:value.value, unit:value.unit, label:value.label};
            value.held = false;
          } else if(held[id]){
            value.value = held[id].value;
            value.unit = held[id].unit;
            value.label = value.label || held[id].label;
            value.held = true;
          }
          return value;
        });
        Object.keys(held).forEach(function(id){
          if(seen[id]) return;
          merged.push({id:id, label:held[id].label, value:held[id].value,
            unit:held[id].unit, present:false, held:true});
        });
        return merged;
      },
      clear: function(){ Object.keys(held).forEach(function(id){ delete held[id]; }); },
    };
  }

  function boundedSenderStats(raw){
    const value = raw && typeof raw === 'object' ? raw : {};
    return {
      type: 'sender_stats',
      sent_fps: Number(finiteNumber(value.sent_fps, 0, 0, 60).toFixed(2)),
      target_fps: Number(finiteNumber(value.target_fps, 0, 0, 60).toFixed(2)),
      width: boundedInteger(value.width, 2, 2, MAX_DIMENSION),
      height: boundedInteger(value.height, 2, 2, MAX_DIMENSION),
      tier: boundedInteger(value.tier, 0, 0, 16),
      quality_tier: boundedInteger(value.quality_tier, 0, 0, 16),
      jpeg_quality: Number(finiteNumber(value.jpeg_quality, 0.92, 0.1, 1).toFixed(2)),
      encode_p90_ms: Number(finiteNumber(value.encode_p90_ms, 0, 0, 60000).toFixed(2)),
      jpeg_bytes: boundedInteger(value.jpeg_bytes, 0, 0, 33554432),
      buffered_bytes: boundedInteger(value.buffered_bytes, 0, 0, 33554432),
      encode_busy_skips: boundedInteger(value.encode_busy_skips, 0, 0, 100000),
      backpressure_skips: boundedInteger(value.backpressure_skips, 0, 0, 100000),
      transport_wait_skips: boundedInteger(value.transport_wait_skips, 0, 0, 100000),
      ack_p90_ms: Number(finiteNumber(value.ack_p90_ms, 0, 0, 60000).toFixed(2)),
      ack_timeouts: boundedInteger(value.ack_timeouts, 0, 0, 100000),
      constrained: value.constrained === true,
    };
  }

  function boundedCaptureProfile(raw){
    const value = raw && typeof raw === 'object' ? raw : {};
    return {
      type: 'capture_profile',
      width: boundedInteger(value.width, 2, 2, MAX_DIMENSION),
      height: boundedInteger(value.height, 2, 2, MAX_DIMENSION),
      tier: boundedInteger(value.tier, 0, 0, 16),
      quality_tier: boundedInteger(value.quality_tier, 0, 0, 16),
      jpeg_quality: Number(finiteNumber(value.jpeg_quality, 0.92, 0.1, 1).toFixed(2)),
    };
  }

  function createCaptureProfileBoundary(){
    let lastKey = null;
    return {
      reset: function(){ lastKey = null; },
      publish: function(raw, state, send){
        const profile = boundedCaptureProfile(raw);
        const key = [profile.width, profile.height, profile.tier,
          profile.quality_tier, profile.jpeg_quality].join(':');
        const options = state && typeof state === 'object' ? state : {};
        if(!options.force && key === lastKey){
          return {sent:false, current:true, deferred:false, profile:profile};
        }
        if(!options.force && (options.encode_in_flight || options.frame_pending)){
          return {sent:false, current:false, deferred:true, profile:profile};
        }
        send(JSON.stringify(profile));
        lastKey = key;
        return {sent:true, current:false, deferred:false, profile:profile};
      },
    };
  }

  async function withEstablishmentGuard(setEstablishing, setup, cleanup){
    setEstablishing(true);
    try{
      return await setup();
    }catch(err){
      setEstablishing(false);
      try{ cleanup(); }catch(_){ /* retain the setup error */ }
      throw err;
    }
  }

  return {
    DEFAULT_CONFIG: DEFAULT_CONFIG,
    JPEG_QUALITIES: JPEG_QUALITIES,
    UPSHIFT_HEALTHY_WINDOWS: UPSHIFT_HEALTHY_WINDOWS,
    normalizeConfig: normalizeConfig,
    configFromStateMessage: configFromStateMessage,
    fitEven: fitEven,
    buildProfiles: buildProfiles,
    nextFrameDeadline: nextFrameDeadline,
    mediaPathHealthy: mediaPathHealthy,
    createFrameFlowController: createFrameFlowController,
    createAdaptivePolicy: createAdaptivePolicy,
    createRecoveryController: createRecoveryController,
    createHeldValueCache: createHeldValueCache,
    boundedSenderStats: boundedSenderStats,
    boundedCaptureProfile: boundedCaptureProfile,
    createCaptureProfileBoundary: createCaptureProfileBoundary,
    withEstablishmentGuard: withEstablishmentGuard,
  };
}));
