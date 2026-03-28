"""Abstract base class for corpus handlers.

A :class:`CorpusHandler` wraps a document corpus and exposes three
capabilities needed by the evaluation pipeline:

1. **Document retrieval** — fetch one or many documents by ``wikipedia_id``.
2. **Decile boundaries** — compute or load pre-computed popularity-decile
   boundary arrays (both unweighted and chunk-weighted flavours).
3. **Metadata persistence** — write computed boundaries to a ``metadata.json``
   file so future runs skip the corpus scan entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from langchain.schema import Document


class CorpusHandler(ABC):
    """Interface every corpus backend must implement.

    All three abstract methods must be provided by concrete subclasses.
    The ``get_boundaries`` method serves as the primary entry point used
    by :class:`~src.question_input.HuggingFaceCyroInput`; it should
    return cached boundaries when available and fall back to a full
    corpus scan otherwise.

    Boundary arrays follow the convention established in
    :mod:`src.metrics.decile_utils`:

    * Shape ``(n_deciles + 1,)`` — ``n_deciles`` internal edges plus the
      global min/max as first/last elements.
    * Unweighted: one count per unique document.
    * Chunk-weighted: one count per chunk produced by splitting the document.
    """

    # ── Document retrieval ─────────────────────────────────────────────────

    @abstractmethod
    def get_documents(self, wikipedia_ids: int | list[int]) -> list[Document]:
        """Return LangChain :class:`~langchain.schema.Document` objects for
        the given Wikipedia article ID(s).

        The ``metadata`` dict of each returned document must include at
        least ``wikipedia_id`` (``int``) and ``popularity_avg`` (``float``).

        Args:
            wikipedia_ids: A single ``wikipedia_id`` or a list of them.

        Returns:
            List of :class:`~langchain.schema.Document` instances,
            one per matched corpus row. Returns an empty list when none
            of the requested IDs are found.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    # ── Decile boundaries ──────────────────────────────────────────────────

    @abstractmethod
    def get_boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(boundaries_uw, boundaries_cw)`` popularity-decile edges.

        Implementations should load pre-computed boundaries from a
        ``metadata.json`` when available, and fall back to a full corpus
        scan via :meth:`create_metadata` (or an equivalent internal scan)
        otherwise.

        Returns:
            A 2-tuple:

            * ``boundaries_uw`` — unweighted boundary array,
              shape ``(n_deciles + 1,)``.
            * ``boundaries_cw`` — chunk-weighted boundary array,
              shape ``(n_deciles + 1,)``.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    # ── Metadata persistence ───────────────────────────────────────────────

    @abstractmethod
    def create_metadata(self, output_path: Path) -> None:
        """Scan the corpus, compute boundaries, and write ``metadata.json``.

        Calling this method must produce a file at ``output_path`` that is
        compatible with
        :func:`~src.metrics.decile_utils.load_boundaries_from_metadata`.
        After this call, :meth:`get_boundaries` should load from the file
        rather than re-scanning.

        Args:
            output_path: Destination path for the ``metadata.json`` file.
                The parent directory will be created if it does not exist.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
