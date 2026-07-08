"""Pipeline: capture -> shared extractors -> scheduled modules -> aggregator."""
from __future__ import annotations

from .camera import Camera
from .context import FrameContext
from .events import Result
from .scheduler import Scheduler


class Pipeline:
    """Orchestrates capture -> extractors -> scheduler -> aggregator -> advisor each frame."""
    def __init__(self, camera: Camera, extractors: list, scheduler: Scheduler,
                 aggregator, advisor_engine=None):
        self.camera = camera
        self.extractors = extractors
        self.scheduler = scheduler
        self.aggregator = aggregator
        self.advisor_engine = advisor_engine

    def process_frame(self, ctx: FrameContext) -> list[Result]:
        """Run extractors, modules, and the advisor for one frame."""
        for ex in self.extractors:
            ex.extract(ctx)
        results = self.scheduler.tick(ctx)
        self.aggregator.ingest(results)
        if self.advisor_engine is not None:
            advice = self.advisor_engine.evaluate(self.aggregator.snapshot())
            if advice:
                self.aggregator.ingest(advice)
                results.extend(advice)
        return results

    def run(self, on_frame=None, max_frames: int | None = None) -> None:
        """on_frame(ctx, results) -> bool; return False to stop."""
        try:
            for ctx in self.camera.frames():
                results = self.process_frame(ctx)
                if on_frame is not None and on_frame(ctx, results) is False:
                    break
                if max_frames is not None and ctx.frame_index + 1 >= max_frames:
                    break
        finally:
            self.camera.release()
            for ex in self.extractors:
                close = getattr(ex, "close", None)
                if close:
                    close()
            for module in self.scheduler.modules:
                close = getattr(module, "close", None)
                if close:
                    close()
