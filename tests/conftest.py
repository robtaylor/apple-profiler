"""Test configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_FIXTURES_DIR = FIXTURES_DIR / "real"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def toc_xml() -> str:
    return (FIXTURES_DIR / "toc.xml").read_text()


@pytest.fixture
def cpu_profile_xml() -> str:
    return (FIXTURES_DIR / "cpu_profile.xml").read_text()


@pytest.fixture
def potential_hangs_xml() -> str:
    return (FIXTURES_DIR / "potential_hangs.xml").read_text()


@pytest.fixture
def os_signpost_xml() -> str:
    return (FIXTURES_DIR / "os_signpost.xml").read_text()


@pytest.fixture
def os_signpost_interval_xml() -> str:
    return (FIXTURES_DIR / "os_signpost_interval.xml").read_text()


# ── Real trace fixtures ──


@pytest.fixture
def real_fixtures_dir() -> Path:
    return REAL_FIXTURES_DIR


@pytest.fixture
def real_finder_toc_xml() -> str:
    return (REAL_FIXTURES_DIR / "finder_toc.xml").read_text()


@pytest.fixture
def real_finder_cpu_profile_xml() -> str:
    return (REAL_FIXTURES_DIR / "finder_cpu_profile.xml").read_text()


@pytest.fixture
def real_hangs_toc_xml() -> str:
    return (REAL_FIXTURES_DIR / "hangs_toc.xml").read_text()


@pytest.fixture
def real_hangs_xml() -> str:
    return (REAL_FIXTURES_DIR / "hangs_potential_hangs.xml").read_text()


@pytest.fixture
def real_signpost_toc_xml() -> str:
    return (REAL_FIXTURES_DIR / "signpost_toc.xml").read_text()


@pytest.fixture
def real_signpost_events_xml() -> str:
    return (REAL_FIXTURES_DIR / "signpost_events.xml").read_text()


@pytest.fixture
def real_signpost_intervals_xml() -> str:
    return (REAL_FIXTURES_DIR / "signpost_intervals.xml").read_text()
