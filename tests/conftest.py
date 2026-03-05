"""Test configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
