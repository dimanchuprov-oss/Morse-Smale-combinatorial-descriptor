import numpy as np
import pytest

from morse_lidar.persistence import PersistenceReport, PersistenceInterval, cubical_persistence
from morse_lidar.ttk_backend import status


def test_persistence_filter_is_deterministic():
    report = PersistenceReport(
        "test",
        (PersistenceInterval(0, 0.0, 0.5), PersistenceInterval(0, 1.0, 1.01)),
    )
    assert len(report.significant(0.1, dimension=0)) == 1


def test_gudhi_backend_has_actionable_optional_dependency_error():
    try:
        import gudhi  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(RuntimeError, match="topology"):
            cubical_persistence(np.zeros((3, 3)))
    else:
        pytest.skip("GUDHI is installed in this environment")


def test_ttk_status_is_explicit():
    result = status()
    assert isinstance(result.available, bool)
    assert result.message
