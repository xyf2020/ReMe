"""Application context: shared state container for components, jobs, and service."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig

if TYPE_CHECKING:
    from .base_component import BaseComponent
    from .job import BaseJob
    from .service import BaseService


class ApplicationContext:
    """Passive state container holding parsed config and wired components.

    The Application class performs the actual wiring (registry lookups and
    component instantiation); this class only stores the results so that
    components, jobs, and the service can find each other at runtime.
    """

    def __init__(self, **kwargs):
        # Parse raw kwargs into a typed, validated config object.
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)

        # Populated by Application during initialization.
        self.service: "BaseService | None" = None
        self.components: dict[ComponentEnum, dict[str, "BaseComponent"]] = {}
        self.jobs: dict[str, "BaseJob"] = {}
        self.thread_pool: ThreadPoolExecutor | None = None

        # Application-lifetime shared state. Values remain available across Job and Step
        # invocations while this Application is running, so Jobs may keep cross-call state here.
        # This is in-memory state, not durable storage; use workspace files or a store when state
        # must survive an Application restart.
        #
        # Per-key monotonic counters organized as a nested tree live here too: each node
        # holds a ``value`` counter and a ``children`` dict keyed by the elements of the
        # ``key`` list passed to :func:`reme.utils.counter.global_counter_next`. Sibling
        # keys have independent counters; the root node acts as the global counter for an
        # empty key. A single lock serializes all tree access and increments.
        self.metadata: dict[str, Any] = {
            "_counter_tree": {"value": 0, "children": {}},
            "_counter_tree_lock": threading.Lock(),
        }
