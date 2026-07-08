"""Weather-aware clothing recommendations."""
from __future__ import annotations

from core.context import FrameContext
from core.events import Severity
from core.registry import register
from modules.base import DetectionModule


_LIGHT = {"tank top", "t-shirt"}
_MEDIUM = {"long sleeve shirt", "hoodie", "sweater"}
_HEAVY = {"jacket", "raincoat", "coat"}


@register("clothing_advice")
class ClothingAdvice(DetectionModule):
    """Weather-aware clothing recommendations."""
    interval = 5.0
    cold_c = 8.0
    cool_c = 15.0
    warm_c = 24.0
    hot_c = 28.0
    rain_mm_threshold = 0.1
    wind_kph_threshold = 25.0
    uv_high = 6.0

    def process(self, ctx: FrameContext):
        """Run this detector on the current frame; return Result(s) or None."""
        weather = ctx.extras.get("weather") or {}
        clothing = ctx.extras.get("clothing") or {}
        temp = weather.get("feels_like_c", weather.get("temperature_c"))
        label = str(clothing.get("clothing") or "...").lower()
        if temp is None:
            return self.result("status", "waiting for weather", 0.0, Severity.INFO, "", ttl=12.0)
        temp = float(temp)
        rain = float(weather.get("rain_mm") or weather.get("precipitation_mm") or 0.0)
        wind = float(weather.get("wind_kph") or 0.0)
        uv = weather.get("uv_index")
        uv = float(uv) if uv is not None else None

        suggestions: list[str] = []
        severity = Severity.INFO
        if label in ("...", "unknown"):
            suggestions.append(f"Weather feels like {temp:.0f}C. Clothing detection is still unavailable.")
        elif temp < self.cold_c and label in (_LIGHT | _MEDIUM):
            suggestions.append(f"It feels like {temp:.0f}C. Consider a coat or warm jacket before going out.")
            severity = Severity.NOTICE
        elif temp < self.cool_c and label in _LIGHT:
            suggestions.append(f"It feels cool ({temp:.0f}C). A sweater or light jacket could help.")
            severity = Severity.NOTICE
        elif temp >= self.hot_c and label in _HEAVY:
            suggestions.append(f"It feels hot ({temp:.0f}C). A {label} may be too warm; consider lighter clothing.")
            severity = Severity.NOTICE
        elif temp >= self.warm_c and label in {"hoodie", "sweater", "coat"}:
            suggestions.append(f"It is warm ({temp:.0f}C). You may be more comfortable in lighter clothing.")
            severity = Severity.INFO

        if rain >= self.rain_mm_threshold and label != "raincoat":
            suggestions.append("Rain is likely now. Consider a raincoat or umbrella.")
            severity = max(severity, Severity.NOTICE, key=lambda s: [Severity.INFO, Severity.NOTICE, Severity.WARNING, Severity.ALERT].index(s))
        if wind >= self.wind_kph_threshold and label in _LIGHT:
            suggestions.append("It is windy. A layer may help if you go outside.")
            severity = Severity.NOTICE
        if uv is not None and uv >= self.uv_high and label == "tank top":
            suggestions.append("UV is high. Consider sunscreen or covering shoulders.")
            severity = Severity.NOTICE

        if not suggestions:
            suggestions.append(f"Clothing looks reasonable for weather that feels like {temp:.0f}C.")

        advice = " ".join(suggestions[:2])
        return [
            self.result("recommendation", advice, 0.6, severity, advice, ttl=20.0),
            self.result("weather_feels_like_c", round(temp, 1), 0.8, Severity.INFO, "", ttl=20.0),
            self.result("detected_clothing", label, float(clothing.get("confidence") or 0.0), Severity.INFO, "", ttl=20.0),
        ]
