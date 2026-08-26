# Extending the system

Recipes for the common ways to grow this codebase. Each points at an existing
file to copy. Read [ARCHITECTURE.md](ARCHITECTURE.md) first for the big picture.
Read the nearest package-level `AGENTS.md` before editing, and use
[OPERATIONS.md](OPERATIONS.md) for current run commands and consent boundaries.

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
It is auto-discovered, so the central registry needs no manual edit. A complete
feature may still require focused tests, dashboard or agent-topic presentation,
replay coverage, and documentation. Add only the layers the requested behavior
actually needs. Emit `Severity.ALERT` for anything a caregiver must know (fall,
unresponsiveness); those flow through deterministic policy in `alerts/`.

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

Topics are **data, not code**: add a `TopicSpec` to the `TOPICS` tuple in
`agent/topics.py`. `Policy._candidates` walks that table, so there is no new
branch to write and nothing to wire up.

```python
TopicSpec(
    topic="my_thing", module="my_thing", key="my_key",
    llm_intent="Gently offer help with X. Never name a cause.",
    fallback="I noticed X — would you like a hand?",   # spoken if no LLM
    priority=60, category="movement",
    gate=Threshold(minimum=0.15, decimals=2)),
```

Before you write it, **open the emitting module and read its `self.result(...)`
call**. Confirm the exact key string, the value type, the severity and the
confidence range. A plausible-looking key that no module publishes produces a
spec that can never fire and no error anywhere — that is exactly how one
corroboration rule sat silent for months.

What each field buys you:

- `gate` — an admission test *and* a bounded rendering: `Threshold`, `IsTrue`,
  `OneOf`, `DictField`, `DictCues`, `NonEmptyList`. Set its bounds to mirror the
  detector's own reporting threshold, so the spec can only narrow what the
  module already chose to publish, never re-derive a finding from a raw number.
- `min_severity` — defaults to `NOTICE`. Only drop it to `INFO` for a signal
  that is genuinely rare (`modules/presence.py` publishes an INFO row every
  frame), and add the pair to `_INFO_JUSTIFIED` in `tests/topic_table_test.py`
  with the reason. A test fails otherwise.
- `category` — the `agent/attention.py` rate-limit bucket. `"general"` has no
  gap and is never category-gated.
- `corroborated_by` — **required** if a `FollowUpRule` in
  `agent/corroboration.py::DEFAULT_RULES` prefix-matches your `(module, key)`;
  a test enforces it. It makes the signal either *asked about* (below the
  rule's `max_confidence`) or *mentioned* (above it), never both.
- `health_prompt` — `None` defers to `kind`. Force `False` for anything that is
  not a health check-in, so it does not spend the hourly check-in budget.

Write `fallback` by hand and keep it warm, non-clinical, and phrased as an offer
or a gentle question — never an assertion or a diagnosis. It is what gets spoken
when the language provider is down, and it is what the safety airlocks in
`agent/corroboration.py::safe_check_in` fall back *to*. `signature` gives
no-repeat behavior; `priority` orders competing topics.

Config may retune a topic but never reword it. Under `conversation.topics` in
`config/modules.yaml` you may set `enabled`, `priority`, `min_confidence`,
`min_quality`, `min_severity`, and a nested `gate:` mapping of numeric bounds.
`TopicSpec.apply_overrides` raises on anything else, so a typo fails at load
rather than silently leaving the deployed spec on its defaults.

ALERTs are **not** handled here — they go through `alerts/`.

Exercise a new topic end to end with the `topic_coverage` fixture in
`config/replay_scenarios.json`:

```powershell
.venv\Scripts\python.exe main.py --source replay:topic_coverage --headless --dev-mode `
  --no-voice --no-moondream --max-frames 600
```

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

```powershell
.venv\Scripts\python.exe tests/smoke_test.py
.venv\Scripts\python.exe -m pytest -q tests/<your_focused_test>.py
interrogate -c pyproject.toml .
graphify update .
```

Add a one-line docstring to every new public class/function — `interrogate` will
fail the build otherwise, which is what keeps the codebase self-documenting.
Use portable `python`/`interrogate` commands instead when the project environment
is already active. Runtime-facing changes also require the bounded verification
described in [AI_DEVELOPMENT.md](AI_DEVELOPMENT.md).
