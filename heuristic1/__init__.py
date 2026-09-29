"""heuristic-1: bonsai 27B thinks, decider judges, MetaCog picks the winner.

Serve the merge as one OpenAI-compatible model:

    python -m heuristic1.server          # :8200, model id "heuristic-1"

Licensing: MIT. The models are Apache-2.0 and are not redistributed here —
Bonsai 2 27B (Prism ML) as thinker, decider-4b (Mapika) as judge. The loop is
MetaCog (MIT). See the repository NOTICE.
"""

from .server import MODEL_ID, build_mc, clean_answer, serve

__version__ = "0.1.0"

__all__ = ["MODEL_ID", "build_mc", "clean_answer", "serve", "__version__"]
