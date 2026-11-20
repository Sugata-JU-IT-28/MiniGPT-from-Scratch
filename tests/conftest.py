"""Keep tiny CPU correctness tests on a single reproducible thread."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_test_threads() -> Iterator[None]:
    """Avoid thread-pool overhead for miniature attention and optimizer fixtures."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
