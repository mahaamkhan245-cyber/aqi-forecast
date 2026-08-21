import requests
import streamlit as st


API_URL = st.secrets["API_URL"].rstrip("/")


def _get(endpoint, timeout=60):
    url = f"{API_URL}/{endpoint.lstrip('/')}"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    return response.json()


def health():
    return _get("health", timeout=90)


def get_forecast():
    return _get("forecast", timeout=90)


def get_features():
    return _get("features", timeout=60)