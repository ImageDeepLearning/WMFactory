const EMPTY_FRAME_DATA_URL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
const CAMERA_DEADZONE = 0.08;
const INVERT_CAMERA_X = true;
const INVERT_CAMERA_Y = true;
const MINEWORLD_INVERT_CAMERA_X = false;
const MINEWORLD_INVERT_CAMERA_Y = false;
const VID2WORLD_INVERT_CAMERA_X = false;
const VID2WORLD_INVERT_CAMERA_Y = false;
const MINEWORLD_CAMERA_DEADZONE = 0.015;
const MINEWORLD_CAMERA_DELTA_GAIN = 0.3;
const MINEWORLD_CAMERA_MAX_DELTA = 0.35;
const SHARED_CONTROL_PROFILES = {
  diamond: { invertX: true, invertY: true },
  matrixgame: { invertX: false, invertY: false },
  "open-oasis": { invertX: false, invertY: false },
  worldfm: { invertX: false, invertY: false },
  mineworld: { invertX: MINEWORLD_INVERT_CAMERA_X, invertY: MINEWORLD_INVERT_CAMERA_Y },
  "infinite-world": { invertX: false, invertY: false },
  vid2world: { invertX: VID2WORLD_INVERT_CAMERA_X, invertY: VID2WORLD_INVERT_CAMERA_Y },
  default: { invertX: INVERT_CAMERA_X, invertY: INVERT_CAMERA_Y },
};

const state = {
  models: [],
  leftModelId: null,
  rightModelId: null,
  leftLoaded: false,
  rightLoaded: false,
  seedImage: null,
  stepping: false,
  sessionActive: false,
  controls: {
    w: false,
    a: false,
    s: false,
    d: false,
    camera_dx: 0,
    camera_dy: 0,
    l_click: false,
    r_click: false,
  },
  mineworldCameraActive: false,
  mineworldCameraPointerId: null,
  mineworldCameraLastX: 0,
  mineworldCameraLastY: 0,
};

const el = {
  leftModelSelect: document.getElementById("leftModelSelect"),
  rightModelSelect: document.getElementById("rightModelSelect"),
  loadArenaBtn: document.getElementById("loadArenaBtn"),
  startBattleBtn: document.getElementById("startBattleBtn"),
  resetBtn: document.getElementById("resetBtn"),
  imageInput: document.getElementById("imageInput"),
  seedPreview: document.getElementById("seedPreview"),
  battleStatus: document.getElementById("battleStatus"),
  leftModelStatus: document.getElementById("leftModelStatus"),
  rightModelStatus: document.getElementById("rightModelStatus"),
  leftFrameView: document.getElementById("leftFrameView"),
  rightFrameView: document.getElementById("rightFrameView"),
  leftLabel: document.getElementById("leftLabel"),
  rightLabel: document.getElementById("rightLabel"),
  leftMeta: document.getElementById("leftMeta"),
  rightMeta: document.getElementById("rightMeta"),
  startOverlay: document.getElementById("startOverlay"),
  startFloatingBtn: document.getElementById("startFloatingBtn"),
  cameraStick: document.getElementById("cameraStick"),
  cameraKnob: document.getElementById("cameraKnob"),
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function setBattleStatus(text, isError = false) {
  el.battleStatus.textContent = text;
  el.battleStatus.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function updateOverlay() {
  el.startOverlay.classList.toggle("hidden", state.sessionActive);
}

function getModelLabel(modelId) {
  return state.models.find((item) => item.id === modelId)?.label || modelId || "未选择";
}

function updateHeadings() {
  el.leftLabel.textContent = getModelLabel(state.leftModelId);
  el.rightLabel.textContent = getModelLabel(state.rightModelId);
}

function setSideLoadedMeta(side, payload) {
  const meta = `${payload.model_id} / GPU ${payload.visible_devices || payload.gpu_index}`;
  if (side === "left") {
    el.leftMeta.textContent = meta;
    el.leftModelStatus.textContent = `已加载到 ${payload.device}`;
  } else {
    el.rightMeta.textContent = meta;
    el.rightModelStatus.textContent = `已加载到 ${payload.device}`;
  }
}

function clearArenaFrames() {
  el.leftFrameView.src = state.seedImage || EMPTY_FRAME_DATA_URL;
  el.rightFrameView.src = state.seedImage || EMPTY_FRAME_DATA_URL;
}

function showSeedEverywhere() {
  if (!state.seedImage) {
    return;
  }
  el.seedPreview.src = state.seedImage;
  clearArenaFrames();
  state.sessionActive = false;
  updateOverlay();
}

function syncDistinctSelects(changedSide) {
  if (state.leftModelId !== state.rightModelId) {
    return;
  }
  const fallback = state.models.find((item) => item.id !== state.leftModelId)?.id;
  if (!fallback) {
    return;
  }
  if (changedSide === "left") {
    state.rightModelId = fallback;
    el.rightModelSelect.value = fallback;
  } else {
    state.leftModelId = fallback;
    el.leftModelSelect.value = fallback;
  }
}

function invalidateLoadedArena(reasonText) {
  state.leftLoaded = false;
  state.rightLoaded = false;
  state.sessionActive = false;
  el.leftModelStatus.textContent = "未加载";
  el.rightModelStatus.textContent = "未加载";
  el.leftMeta.textContent = "等待加载";
  el.rightMeta.textContent = "等待加载";
  updateOverlay();
  if (reasonText) {
    setBattleStatus(reasonText);
  }
}

async function loadModels() {
  const data = await api("/api/models");
  state.models = data.models || [];

  for (const select of [el.leftModelSelect, el.rightModelSelect]) {
    select.innerHTML = "";
    for (const model of state.models) {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = model.label;
      select.appendChild(opt);
    }
  }

  state.leftModelId = state.models[0]?.id || null;
  state.rightModelId = state.models[1]?.id || state.models[0]?.id || null;
  if (state.leftModelId) {
    el.leftModelSelect.value = state.leftModelId;
  }
  if (state.rightModelId) {
    el.rightModelSelect.value = state.rightModelId;
  }
  updateHeadings();
}

async function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function onLoadArena() {
  if (!state.leftModelId || !state.rightModelId) {
    setBattleStatus("模型列表为空，无法启动 WMArena。", true);
    return;
  }
  if (state.leftModelId === state.rightModelId) {
    setBattleStatus("第一阶段要求左右必须选择两个不同模型。", true);
    return;
  }

  setBattleStatus("正在为左右两侧分配独立 GPU 并加载模型...");
  el.leftModelStatus.textContent = "加载中...";
  el.rightModelStatus.textContent = "加载中...";
  try {
    const data = await api("/api/arena/load", {
      method: "POST",
      body: JSON.stringify({
        left_model_id: state.leftModelId,
        right_model_id: state.rightModelId,
      }),
    });
    state.leftLoaded = true;
    state.rightLoaded = true;
    setSideLoadedMeta("left", data.left);
    setSideLoadedMeta("right", data.right);
    setBattleStatus("双模型已就绪。上传相同初始图后即可开始对战。");
  } catch (err) {
    state.leftLoaded = false;
    state.rightLoaded = false;
    el.leftMeta.textContent = "加载失败";
    el.rightMeta.textContent = "加载失败";
    setBattleStatus(`加载失败: ${err.message}`, true);
  }
}

async function onStartBattle() {
  if (!state.seedImage) {
    setBattleStatus("请先上传一张共同的初始图像。", true);
    return;
  }
  if (!state.leftLoaded || !state.rightLoaded) {
    setBattleStatus("请先完成双模型加载。", true);
    return;
  }

  setBattleStatus("正在用同一张图初始化左右会话...");
  try {
    const data = await api("/api/arena/start", {
      method: "POST",
      body: JSON.stringify({ init_image_base64: state.seedImage }),
    });
    state.sessionActive = true;
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    updateOverlay();
    setBattleStatus("对战已开始。现在所有控制输入都会同步广播到两侧。");
  } catch (err) {
    setBattleStatus(`启动失败: ${err.message}`, true);
  }
}

async function onResetBattle() {
  if (!state.sessionActive) {
    return;
  }
  try {
    const data = await api("/api/arena/reset", {
      method: "POST",
      body: JSON.stringify({ init_image_base64: state.seedImage }),
    });
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;
    setBattleStatus("左右会话已同步重置。");
  } catch (err) {
    setBattleStatus(`重置失败: ${err.message}`, true);
  }
}

function isChunkedModel(modelId) {
  return ["yume", "infinite-world", "gamecraft", "worldplay", "lingbot-world"].includes(modelId);
}

function isLatencyModel(modelId) {
  return modelId === "vid2world";
}

function chunkedModelLabel(modelId) {
  const labels = {
    "infinite-world": "Infinite-World",
    yume: "YUME",
    gamecraft: "GameCraft",
    worldplay: "WorldPlay",
    "lingbot-world": "LingBot-World",
  };
  return labels[modelId] || "Chunked Model";
}

function controlModelId() {
  return state.leftModelId;
}

async function stepLoop() {
  if (!state.sessionActive || state.stepping) {
    return;
  }

  const modelId = controlModelId();
  const cameraDeadzone = modelId === "mineworld" ? MINEWORLD_CAMERA_DEADZONE : CAMERA_DEADZONE;
  if (Math.abs(state.controls.camera_dx) <= cameraDeadzone) {
    state.controls.camera_dx = 0;
  }
  if (Math.abs(state.controls.camera_dy) <= cameraDeadzone) {
    state.controls.camera_dy = 0;
  }

  const hasInput =
    !!state.controls.w ||
    !!state.controls.a ||
    !!state.controls.s ||
    !!state.controls.d ||
    !!state.controls.l_click ||
    !!state.controls.r_click ||
    Math.abs(state.controls.camera_dx) > cameraDeadzone ||
    Math.abs(state.controls.camera_dy) > cameraDeadzone;

  if (!hasInput) {
    return;
  }

  state.stepping = true;
  const action = { ...state.controls };
  try {
    if (isChunkedModel(modelId)) {
      setBattleStatus(`${chunkedModelLabel(modelId)} 控制映射已应用，正在等待左右两侧都完成当前生成...`);
    }
    const data = await api("/api/arena/step", {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    el.leftFrameView.src = `data:image/png;base64,${data.left.frame_base64}`;
    el.rightFrameView.src = `data:image/png;base64,${data.right.frame_base64}`;

    const leftLatency = Number(data?.left?.extra?.latency_ms || 0);
    const rightLatency = Number(data?.right?.extra?.latency_ms || 0);
    if (leftLatency > 0 || rightLatency > 0 || isLatencyModel(state.leftModelId) || isLatencyModel(state.rightModelId)) {
      const leftSeconds = leftLatency > 0 ? `${(leftLatency / 1000).toFixed(1)}s` : "-";
      const rightSeconds = rightLatency > 0 ? `${(rightLatency / 1000).toFixed(1)}s` : "-";
      setBattleStatus(`同步 step 完成。左侧耗时 ${leftSeconds}，右侧耗时 ${rightSeconds}。`);
    }

    if (data.left.ended || data.left.truncated || data.right.ended || data.right.truncated) {
      setBattleStatus("至少一侧回合结束，正在同步重置。", true);
      await onResetBattle();
    }
  } catch (err) {
    setBattleStatus(`Step 失败: ${err.message}`, true);
  } finally {
    if (modelId === "mineworld") {
      state.controls.camera_dx = 0;
      state.controls.camera_dy = 0;
      paintMineWorldCamera(0, 0);
    }
    state.stepping = false;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cameraInversionForModel() {
  const modelId = controlModelId();
  return SHARED_CONTROL_PROFILES[modelId] || SHARED_CONTROL_PROFILES.default;
}

function paintMineWorldCamera(dx, dy) {
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;
  const x = center - knobSize / 2 + dx * maxRadius;
  const y = center - knobSize / 2 + dy * maxRadius;
  el.cameraKnob.style.left = `${x}px`;
  el.cameraKnob.style.top = `${y}px`;
}

function updateWASDStyles() {
  document.querySelectorAll(".key").forEach((btn) => {
    const key = btn.dataset.key;
    btn.classList.toggle("active", !!state.controls[key]);
  });
}

function bindKeyboard() {
  const downMap = { w: "w", a: "a", s: "s", d: "d" };

  window.addEventListener("keydown", (e) => {
    const key = downMap[e.key.toLowerCase()];
    if (!key) {
      return;
    }
    state.controls[key] = true;
    updateWASDStyles();
  });

  window.addEventListener("keyup", (e) => {
    const key = downMap[e.key.toLowerCase()];
    if (!key) {
      return;
    }
    state.controls[key] = false;
    updateWASDStyles();
  });

  window.addEventListener("blur", () => {
    state.controls.w = false;
    state.controls.a = false;
    state.controls.s = false;
    state.controls.d = false;
    state.controls.l_click = false;
    state.controls.r_click = false;
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    state.mineworldCameraActive = false;
    state.mineworldCameraPointerId = null;
    paintMineWorldCamera(0, 0);
    updateWASDStyles();
  });
}

function bindWASDButtons() {
  document.querySelectorAll(".key").forEach((btn) => {
    const key = btn.dataset.key;
    const press = () => {
      state.controls[key] = true;
      updateWASDStyles();
    };
    const release = () => {
      state.controls[key] = false;
      updateWASDStyles();
    };
    btn.addEventListener("pointerdown", press);
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("pointerleave", release);
  });
}

function bindCameraStick() {
  const stick = el.cameraStick;
  const knob = el.cameraKnob;
  const size = 160;
  const knobSize = 54;
  const center = size / 2;
  const maxRadius = (size - knobSize) / 2;
  let active = false;

  const paint = (dx, dy) => {
    const x = center - knobSize / 2 + dx * maxRadius;
    const y = center - knobSize / 2 + dy * maxRadius;
    knob.style.left = `${x}px`;
    knob.style.top = `${y}px`;
  };

  const setFromPointer = (clientX, clientY) => {
    const rect = stick.getBoundingClientRect();
    const x = clientX - rect.left - center;
    const y = clientY - rect.top - center;
    const len = Math.hypot(x, y);
    const scale = len > maxRadius ? maxRadius / len : 1;

    const nx = (x * scale) / maxRadius;
    const ny = (y * scale) / maxRadius;
    const inv = cameraInversionForModel();
    state.controls.camera_dx = Number((inv.invertX ? -nx : nx).toFixed(3));
    state.controls.camera_dy = Number((inv.invertY ? -ny : ny).toFixed(3));
    paint(nx, ny);
  };

  const resetStick = () => {
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    paint(0, 0);
  };

  const resetMineWorldStick = () => {
    state.controls.camera_dx = 0;
    state.controls.camera_dy = 0;
    state.mineworldCameraActive = false;
    state.mineworldCameraPointerId = null;
    paintMineWorldCamera(0, 0);
  };

  const updateMineWorldFromDelta = (deltaX, deltaY) => {
    const nx = clamp((deltaX / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const ny = clamp((deltaY / maxRadius) * MINEWORLD_CAMERA_DELTA_GAIN, -MINEWORLD_CAMERA_MAX_DELTA, MINEWORLD_CAMERA_MAX_DELTA);
    const inv = cameraInversionForModel();
    state.controls.camera_dx = Number((inv.invertX ? -nx : nx).toFixed(3));
    state.controls.camera_dy = Number((inv.invertY ? -ny : ny).toFixed(3));
    paintMineWorldCamera(nx, ny);
  };

  stick.addEventListener("pointerdown", (e) => {
    if (controlModelId() === "mineworld") {
      state.mineworldCameraActive = true;
      state.mineworldCameraPointerId = e.pointerId;
      state.mineworldCameraLastX = e.clientX;
      state.mineworldCameraLastY = e.clientY;
      stick.setPointerCapture?.(e.pointerId);
      return;
    }
    active = true;
    setFromPointer(e.clientX, e.clientY);
  });

  window.addEventListener("pointermove", (e) => {
    if (controlModelId() === "mineworld") {
      if (!state.mineworldCameraActive || state.mineworldCameraPointerId !== e.pointerId) {
        return;
      }
      const deltaX = e.clientX - state.mineworldCameraLastX;
      const deltaY = e.clientY - state.mineworldCameraLastY;
      state.mineworldCameraLastX = e.clientX;
      state.mineworldCameraLastY = e.clientY;
      updateMineWorldFromDelta(deltaX, deltaY);
      return;
    }
    if (!active) {
      return;
    }
    setFromPointer(e.clientX, e.clientY);
  });

  const end = (e) => {
    if (controlModelId() === "mineworld") {
      if (!state.mineworldCameraActive) {
        return;
      }
      if (e && state.mineworldCameraPointerId !== null && e.pointerId !== state.mineworldCameraPointerId) {
        return;
      }
      resetMineWorldStick();
      return;
    }
    if (!active) {
      return;
    }
    active = false;
    resetStick();
  };

  window.addEventListener("pointerup", end);
  window.addEventListener("pointercancel", end);
  resetStick();
}

function bindEvents() {
  el.leftModelSelect.addEventListener("change", () => {
    state.leftModelId = el.leftModelSelect.value;
    syncDistinctSelects("left");
    updateHeadings();
    invalidateLoadedArena("模型选择已变化，请重新加载双模型。");
  });

  el.rightModelSelect.addEventListener("change", () => {
    state.rightModelId = el.rightModelSelect.value;
    syncDistinctSelects("right");
    updateHeadings();
    invalidateLoadedArena("模型选择已变化，请重新加载双模型。");
  });

  el.imageInput.addEventListener("change", async (e) => {
    const [file] = e.target.files || [];
    if (!file) {
      return;
    }
    state.seedImage = await readFileAsDataUrl(file);
    showSeedEverywhere();
    setBattleStatus(`已设置共同初始图像: ${file.name}`);
  });

  el.loadArenaBtn.addEventListener("click", onLoadArena);
  el.startBattleBtn.addEventListener("click", onStartBattle);
  el.startFloatingBtn.addEventListener("click", onStartBattle);
  el.resetBtn.addEventListener("click", onResetBattle);

  bindKeyboard();
  bindWASDButtons();
  bindCameraStick();
}

async function boot() {
  try {
    await loadModels();
    bindEvents();
    el.seedPreview.src = EMPTY_FRAME_DATA_URL;
    el.leftFrameView.src = EMPTY_FRAME_DATA_URL;
    el.rightFrameView.src = EMPTY_FRAME_DATA_URL;
    updateOverlay();
    setInterval(stepLoop, 80);
    setBattleStatus("就绪：左右选择两个不同模型，上传共同初始图，然后开始对战。");
  } catch (err) {
    setBattleStatus(`初始化失败: ${err.message}`, true);
  }
}

boot();
