# Extending the system

Recipes for the common ways to grow this codebase. Each points at an existing
file to copy. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for the big picture.

Golden rules:
- Modules only **read** `FrameContext` and **return** `Result`s — no side effects
  on other modules except via `ctx.extras`.
- Anything slow (model inference, network, TTS) runs **off the video loop** (a
  `ThreadPoolExecutor` + cached result, a subprocess, or a daemon thread).
- Optional dependencies **degrade gracefully**: lazy-import, set an `available`
  flag, fall back. Secrets come from `.env`, never yaml.
- Health signals are screening prompts, not diagnoses; anything urgent must flow
  through the deterministic `alerts/` path, not the LLM.

## 1. Add a detection module

Create `modules/my_thing.py` (copy any small module, e.g. `modules/balance.py`):

```python
from core.registry import register
from core.events import Severity
from modules.base import DetectionModule

@register("my_thing")                 # the name used in config/modules.yaml
class MyThing(DetectionModule):
    interval = 1.0                    # run at most once a second (0 = every frame)
    requires = ("face",)              # skipped unless ctx.face is present
    my_threshold = 0.5                # config-tunable (see recipe 6)

    def process(self, ctx):
        px = ctx.face_px()            # helper: face landmarks in pixels
        value = ...                   # compute something
        if value < self.my_threshold:
            return None               # nothing to report this frame
        return self.result("my_key", round(value, 2), confidence=0.6,
                            severity=Severity.NOTICE, message="Noticed X")
```

Then enable it in `config/modules.yaml`:
```yaml
  my_thing:
    enabled: true
    my_threshold: 0.4
```
It is auto-discovered — no other file changes. Emit `Severity.ALERT` for
anything a caregiver must know (fall, unresponsiveness); those flow to `alerts/`.

## 2. Add a backend (rPPG / emotion "show both")

Backends let one detector run a heuristic **and** tested models side by side.
Copy `modules/rppg_backends/classical.py` or `modules/emotion_backends/heuristic.py`:

```python
from modules.backends.base import Backend

class MyModelBackend(Backend):
    label = "mymodel"                 # shown in result keys/dashboard

    def __init__(self, **params):
        self.available = False
        try:
            import somelib             # lazy import
            self._model = somelib.load()
            self.available = True
        except Exception as e:
            print(f"[mybackend] unavailable ({e})")

    def update(self, ctx):            # cheap: stash what compute() needs
        self._crop = ctx.face.crop if ctx.face else None

    def compute(self):               # return a reading dict or None
        if not self.available or self._crop is None:
            return None
        return {"emotion": ..., "confidence": ...}
```

Register it in the owning module's `_BACKENDS` dict (see `modules/emotion.py`)
and add its label to that module's `backends:` list in the config. Heavy models:
inference on a `ThreadPoolExecutor` with a `_pending` future (see
`rppg_backends/openrppg.py`). The dashboard shows every backend automatically.

## 3. Add a caregiver alert channel

Copy a class in `alerts/notifier.py` (e.g. `WebhookChannel`):

```python
class MyChannel(Channel):
    name = "mychannel"
    def __init__(self):
        self.token = os.environ.get("MYCHANNEL_TOKEN")   # from .env
        self.available = bool(self.token)
    def send(self, subject, body):
        if not self.available:
            return False
        ...  # deliver; return True on success
```

Add it to `_REGISTRY` at the bottom of `notifier.py`, then list it under
`channels:` in `config/alerts.yaml`. Missing creds → the channel self-disables;
`console` is always kept so alerts are never lost.

## 4. Add a voice-agent conversation topic

The agent decides what to say in `agent/policy.py: Policy._candidates`. Append a
candidate `Intent` that reads the `ObservationMemory`:

```python
r = mem.get("my_thing", "my_key")
if r is not None and _ORDER[r.severity] >= _ORDER[Severity.NOTICE] and self._fresh("mything", now):
    cands.append(Intent(
        kind="observation", signature="mything",
        llm_intent="Gently mention X and offer help.",
        detail=str(r.message),
        fallback="I noticed X — are you okay?",   # spoken if no LLM
        priority=60))
```

`signature` gives no-repeat behavior; `priority` orders competing topics; Gemini
phrases `llm_intent`+`detail`, falling back to `fallback` offline. ALERTs are
**not** handled here — they go through `alerts/`.

## 5. Add a dashboard / `/data` field

`output/dashboard.py` is the single source of truth for both the window and the
web `/data` payload. To surface a new value:
- If it's a per-metric backend comparison, nothing to do — any
  `"<metric>_<backend>"` key is picked up by `_parse_comparisons`.
- For a stat grid entry, add `(module, key, "Label")` to `_FATIGUE_STATS` or
  `_CLOTHING_WEATHER_STATS`. Both `render()` (window) and `to_payload()` (`/data`)
  use these lists, so the window and web stay in sync.
- Add the row's markup in `webui/data.html` if it's a new section.

## 6. Add a config key

Add a class attribute with a default on the module/backend; `build_enabled`
passes matching yaml keys into `__init__` (via `DetectionModule.__init__`), so
`self.my_threshold` just works. Document the key inline in `config/modules.yaml`.

## Before you commit — verification loop

```bash
python tests/smoke_test.py                 # whole pipeline on synthetic frames
python tests/<your_focused_test>.py        # e.g. hr_backends_test, agent_alert_test
interrogate -c pyproject.toml .            # docstring coverage must stay >= 95%
graphify update .                          # refresh the knowledge graph (AST-only)
```

Add a one-line docstring to every new public class/function — `interrogate` will
fail the build otherwise, which is what keeps the codebase self-documenting.
