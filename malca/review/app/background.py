# This file was mechanically split from malca.review.app; preserve behavior when editing.
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be.*leaked semaphore",
    module="multiprocessing.resource_tracker",
)

def _configure_background_start_methods() -> None:
    """Prefer spawn so background workers do not inherit the dev-server socket."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        multiprocess = importlib.import_module("multiprocess")
    except ModuleNotFoundError:
        return

    try:
        methods = set(multiprocess.get_all_start_methods())
    except Exception:
        methods = set()

    if "spawn" not in methods:
        return

    try:
        multiprocess.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


_configure_background_start_methods()



CLASS_BADGE_TAGS = list(CLASS_KEY_MAP.values())
TAXONOMY_KEYBOARD_PAYLOAD = keyboard_payload()
MORPHOLOGY_PRIMARY_TAGS = [str(item["value"]) for item in MORPHOLOGY_PRIMARY]

class TrackingDiskcacheManager(DiskcacheManager):
    """Diskcache manager that can clean up outstanding worker processes on exit."""

    def __init__(self, cache=None, cache_by=None, expire=None):
        super().__init__(cache=cache, cache_by=cache_by, expire=expire)
        self._active_jobs: set[int] = set()

    def call_job_fn(self, key, job_fn, args, context):
        job = super().call_job_fn(key, job_fn, args, context)
        if job is not None:
            try:
                self._active_jobs.add(int(job))
            except Exception:
                pass
        return job

    def terminate_job(self, job):
        try:
            return super().terminate_job(job)
        finally:
            try:
                if job is not None:
                    self._active_jobs.discard(int(job))
            except Exception:
                pass

    def get_result(self, key, job):
        try:
            return super().get_result(key, job)
        finally:
            try:
                if job is not None:
                    self._active_jobs.discard(int(job))
            except Exception:
                pass

    def terminate_all_jobs(self) -> None:
        for job in tuple(sorted(self._active_jobs)):
            self.terminate_job(job)


# Background callback manager for long-running fetch/import (DiskCache for local dev)
_bc_cache = diskcache.Cache(_APP_REPO_ROOT / DEFAULT_OUTPUT_DIR / "review" / ".dash_cache")
_background_callback_manager = TrackingDiskcacheManager(_bc_cache)
# High-frequency review UI callbacks (plot render, auto period, sidebar hydration)
# are intentionally synchronous to avoid long-session file-descriptor exhaustion.
_UI_BACKGROUND_CALLBACKS = False
_PRELOAD_DELAY_SEC = 0.4
_PRELOAD_LOOKAHEAD = 1
_preload_generation_lock = Lock()
_preload_generation = 0


def _next_preload_generation() -> int:
    """Return a new monotonic preload generation token."""
    global _preload_generation
    with _preload_generation_lock:
        _preload_generation += 1
        return _preload_generation


def _is_current_preload_generation(generation: int) -> bool:
    """Check whether a queued preload request is still current."""
    with _preload_generation_lock:
        return generation == _preload_generation


def _cleanup_background_resources(*_args) -> None:
    """Terminate outstanding background jobs so Ctrl-C fully releases the port."""
    try:
        if _background_callback_manager is not None:
            _background_callback_manager.terminate_all_jobs()
    except Exception:
        pass
    try:
        _bc_cache.close()
    except Exception:
        pass


atexit.register(_cleanup_background_resources)

# Initialize Dash app
