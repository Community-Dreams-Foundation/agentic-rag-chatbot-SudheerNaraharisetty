"""
Open-Meteo Tools: Weather data retrieval and analysis.
Provides NLP query parsing, geocoding, and statistical analysis.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import requests
from datetime import datetime, timedelta

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """Client for Open-Meteo weather API (free, no key needed)."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.open_meteo_base_url
        self.geolocator = Nominatim(user_agent="agentic-rag-chatbot")

    def geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Convert location name to (latitude, longitude).

        Args:
            location_name: City or place name (e.g., "New York", "London")

        Returns:
            (latitude, longitude) tuple or None if not found
        """
        try:
            location = self.geolocator.geocode(location_name, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Geocoding error for '{location_name}': {e}")
            return None

    def get_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        hourly: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get current/forecast weather data.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            hourly: List of hourly variables to fetch

        Returns:
            Weather data dict from Open-Meteo
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
        }

        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if hourly:
            params["hourly"] = ",".join(hourly)

        response = requests.get(
            f"{self.base_url}/forecast", params=params, timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        daily: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get historical weather data from archive API.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            daily: List of daily variables to fetch

        Returns:
            Historical weather data dict
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
        }

        if daily:
            params["daily"] = ",".join(daily)

        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()


class WeatherAnalyzer:
    """Statistical analysis engine for weather time series data."""

    def __init__(self):
        self.client = OpenMeteoClient()

    @staticmethod
    def _extract_time_series(
        data: Dict[str, Any], variable: str
    ) -> Tuple[List[Any], List[str]]:
        """
        Extract time series values from weather data, handling both
        hourly and daily response formats.

        Returns:
            (values_list, times_list) — may be empty if variable not found
        """
        # Try hourly first (forecast API returns this)
        hourly = data.get("hourly", {})
        if variable in hourly:
            values = hourly.get(variable, [])
            times = hourly.get("time", [])
            return values, times

        # Try daily (archive/historical API returns this)
        daily = data.get("daily", {})
        if variable in daily:
            values = daily.get(variable, [])
            times = daily.get("time", [])
            return values, times

        return [], []

    def analyze_time_series(
        self, data: Dict[str, Any], variable: str = "temperature_2m"
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on weather time series data.

        Works with both hourly (forecast) and daily (historical) data.

        Args:
            data: Weather data from Open-Meteo API
            variable: Variable name to analyze

        Returns:
            Analysis results dict with stats, or error message
        """
        values, times = self._extract_time_series(data, variable)

        if not values:
            return {"error": f"No data found for variable: {variable}"}

        import statistics

        # Filter out None values
        clean_values = [v for v in values if v is not None]

        if not clean_values:
            return {"error": "No valid data points after filtering nulls"}

        analysis = {
            "variable": variable,
            "count": len(clean_values),
            "mean": round(statistics.mean(clean_values), 2),
            "median": round(statistics.median(clean_values), 2),
            "stdev": round(statistics.stdev(clean_values), 2) if len(clean_values) > 1 else 0,
            "min": round(min(clean_values), 2),
            "max": round(max(clean_values), 2),
            "range": round(max(clean_values) - min(clean_values), 2),
        }

        # Volatility (coefficient of variation)
        if analysis["mean"] != 0:
            analysis["volatility"] = round(
                analysis["stdev"] / abs(analysis["mean"]), 4
            )

        # Data quality — missingness check
        total = len(values)
        missing = total - len(clean_values)
        analysis["missing_count"] = missing
        analysis["missing_percent"] = round((missing / total) * 100, 2) if total else 0

        # Time range
        if times:
            analysis["time_start"] = times[0]
            analysis["time_end"] = times[-1]

        return analysis

    def detect_anomalies(
        self,
        data: Dict[str, Any],
        variable: str = "temperature_2m",
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Z-score method.

        Args:
            data: Weather data from Open-Meteo
            variable: Variable to check
            threshold: Z-score threshold for anomaly detection

        Returns:
            Anomaly detection results
        """
        values, times = self._extract_time_series(data, variable)

        if not values:
            return {"error": f"No data found for variable: {variable}"}

        import statistics

        clean_values = [v for v in values if v is not None]
        if len(clean_values) < 2:
            return {"error": "Insufficient data points for anomaly detection"}

        mean = statistics.mean(clean_values)
        stdev = statistics.stdev(clean_values)

        if stdev == 0:
            return {"anomalies": [], "message": "No variation in data (stdev=0)"}

        anomalies = []
        for i, value in enumerate(values):
            if value is not None:
                z_score = (value - mean) / stdev
                if abs(z_score) > threshold:
                    anomalies.append({
                        "time": times[i] if i < len(times) else None,
                        "value": round(value, 2),
                        "z_score": round(z_score, 2),
                        "direction": "high" if z_score > 0 else "low",
                    })

        return {
            "variable": variable,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:10],
            "threshold": threshold,
            "mean": round(mean, 2),
            "stdev": round(stdev, 2),
        }

    def rolling_average(
        self,
        data: Dict[str, Any],
        variable: str = "temperature_2m",
        window: int = 24,
    ) -> Dict[str, Any]:
        """
        Calculate rolling average over the time series.

        Args:
            data: Weather data from Open-Meteo
            variable: Variable to analyze
            window: Rolling window size (data points)

        Returns:
            Rolling average results
        """
        values, times = self._extract_time_series(data, variable)

        if not values:
            return {"error": f"No data found for variable: {variable}"}

        rolling = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = [v for v in values[start : i + 1] if v is not None]
            if window_values:
                avg = sum(window_values) / len(window_values)
                rolling.append({
                    "time": times[i] if i < len(times) else None,
                    "value": values[i],
                    "rolling_avg": round(avg, 2),
                })

        return {
            "variable": variable,
            "window_size": window,
            "data_points": len(rolling),
            "rolling_averages": rolling[-20:],  # Last 20 for display
        }


class WeatherQueryParser:
    """Parse natural language weather queries into structured API parameters."""

    METRIC_KEYWORDS = {
        "temperature": "temperature_2m",
        "temp": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "precipitation": "precipitation",
        "rain": "precipitation",
        "wind": "wind_speed_10m",
        "pressure": "surface_pressure",
    }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language weather query.

        Args:
            query: e.g., "What's the humidity in New York last week?"

        Returns:
            Dict with location, metric, time_period, comparison
        """
        query_lower = query.lower()

        # Extract location — look for "in [Location]" or "for [Location]"
        location = None
        location_patterns = [
            r"in\s+([A-Za-z\s,]+?)(?:\s+last|\s+yesterday|\s+today|\s+now|\s+compared|[?.]|$)",
            r"for\s+([A-Za-z\s,]+?)(?:\s+last|\s+yesterday|\s+today|\s+now|\s+compared|[?.]|$)",
            r"(?:weather|temperature|humidity|rain|wind)\s+(?:in|at|for)\s+([A-Za-z\s,]+?)(?:\s+last|\s+yesterday|[?.]|$)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip().rstrip(",. ")
                break

        # Extract metric
        metric = "temperature_2m"  # default
        for keyword, api_metric in self.METRIC_KEYWORDS.items():
            if keyword in query_lower:
                metric = api_metric
                break

        # Extract time period
        time_period = "current"
        if "tomorrow" in query_lower:
            time_period = "tomorrow"
        elif "today" in query_lower:
            time_period = "today"
        elif "this week" in query_lower or "next few days" in query_lower:
            time_period = "this_week"
        elif "next week" in query_lower:
            time_period = "next_week"
        elif "last week" in query_lower or "past week" in query_lower:
            time_period = "last_week"
        elif "yesterday" in query_lower:
            time_period = "yesterday"
        elif "last month" in query_lower or "past month" in query_lower:
            time_period = "last_month"
        elif "forecast" in query_lower:
            time_period = "this_week"

        # Check for comparison intent
        comparison = (
            "compared" in query_lower
            or "vs" in query_lower
            or "versus" in query_lower
        )

        return {
            "location": location,
            "metric": metric,
            "time_period": time_period,
            "comparison": comparison,
            "original_query": query,
        }

    @staticmethod
    def get_date_range(time_period: str) -> Tuple[str, str]:
        """
        Convert time period keyword to (start_date, end_date) strings.

        Returns:
            Tuple of (YYYY-MM-DD, YYYY-MM-DD)
        """
        today = datetime.now()

        if time_period == "tomorrow":
            tmrw = today + timedelta(days=1)
            return (tmrw.strftime("%Y-%m-%d"), tmrw.strftime("%Y-%m-%d"))
        elif time_period == "today":
            d = today.strftime("%Y-%m-%d")
            return (d, d)
        elif time_period == "this_week":
            end = today + timedelta(days=7)
            return (today.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif time_period == "next_week":
            start = today + timedelta(days=7)
            end = today + timedelta(days=14)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        elif time_period == "yesterday":
            start = today - timedelta(days=1)
            return (start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        elif time_period == "last_week":
            start = today - timedelta(days=7)
            return (start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        elif time_period == "last_month":
            start = today - timedelta(days=30)
            return (start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
        else:  # current
            d = today.strftime("%Y-%m-%d")
            return (d, d)


# ── Agent Tool Interface ──────────────────────────────────────────

def get_weather_for_agent(
    location: str,
    metric: str = "temperature_2m",
    period: str = "current",
) -> Dict[str, Any]:
    """
    Unified weather tool for the agentic orchestrator.

    Called by the agent as:
      {"tool": "get_weather", "args": {"location": "...", "metric": "...", "period": "..."}}

    Args:
        location: City or place name (e.g., "San Francisco")
        metric: Weather variable (temperature_2m, relative_humidity_2m, etc.)
        period: current | yesterday | last_week | last_month

    Returns:
        Dict with location_info, analysis, and optional anomalies
    """
    client = OpenMeteoClient()
    analyzer = WeatherAnalyzer()
    parser = WeatherQueryParser()

    # Geocode location
    coords = client.geocode_location(location)
    if not coords:
        return {"error": f"Could not find location: '{location}'"}

    lat, lon = coords

    result = {
        "location": location,
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "metric": metric,
        "period": period,
    }

    # Determine if future (forecast API) or past (historical/archive API)
    future_periods = {"current", "today", "tomorrow", "this_week", "next_week"}
    past_periods = {"yesterday", "last_week", "last_month"}

    try:
        if period in future_periods:
            # Forecast API — hourly data (handles today + up to 16 days ahead)
            start_date, end_date = parser.get_date_range(period)
            data = client.get_weather(
                lat, lon,
                start_date=start_date,
                end_date=end_date,
                hourly=[metric],
            )
            analysis = analyzer.analyze_time_series(data, variable=metric)
            result["analysis"] = analysis
            result["date_range"] = {"start": start_date, "end": end_date}
        elif period in past_periods:
            # Historical API — daily data
            start_date, end_date = parser.get_date_range(period)
            data = client.get_historical_weather(
                lat, lon, start_date, end_date, daily=[metric]
            )
            analysis = analyzer.analyze_time_series(data, variable=metric)
            anomalies = analyzer.detect_anomalies(data, variable=metric)
            result["analysis"] = analysis
            result["anomalies"] = anomalies
            result["date_range"] = {"start": start_date, "end": end_date}
        else:
            # Fallback: treat as forecast
            data = client.get_weather(lat, lon, hourly=[metric])
            analysis = analyzer.analyze_time_series(data, variable=metric)
            result["analysis"] = analysis

    except requests.RequestException as e:
        result["error"] = f"Weather API request failed: {e}"
    except Exception as e:
        result["error"] = f"Weather analysis failed: {e}"

    return result
