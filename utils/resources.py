"""Cached loaders for Streamlit pages, with friendly error handling."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
import streamlit as st

from utils.data_loader import DataError, load_dataset, profile_dataset
from utils.model_utils import ModelError, ensure_artifacts

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Load and validate the dataset once per session."""
    return load_dataset()


@st.cache_data(show_spinner=False)
def get_dataset_profile() -> dict[str, Any]:
    """Data-quality profile of the current dataset."""
    return profile_dataset(get_dataset())


@st.cache_resource(show_spinner="Preparing the model. This only happens on first run.")
def get_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the saved pipeline and report, training them first if necessary."""
    return ensure_artifacts()


def _load_or_stop(loader: Callable[[], Any], description: str) -> Any:
    try:
        return loader()
    except (DataError, ModelError) as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - never show a traceback to the user
        logger.exception("Unexpected error while loading %s", description)
        st.error(f"The {description} could not be loaded. Check the terminal for details.")
    st.stop()


def load_dataset_or_stop() -> pd.DataFrame:
    """Return the dataset, or show an error and halt the page."""
    return _load_or_stop(get_dataset, "dataset")


def load_dataset_profile_or_stop() -> dict[str, Any]:
    """Return the data profile, or show an error and halt the page."""
    return _load_or_stop(get_dataset_profile, "dataset")


def load_model_or_stop() -> tuple[Any, dict[str, Any]]:
    """Return (pipeline, report), or show an error and halt the page."""
    return _load_or_stop(get_artifacts, "model")
