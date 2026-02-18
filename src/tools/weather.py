"""
Open-Meteo Tools: Weather data retrieval and analysis.
Implements safe sandbox for data analysis.
"""

import json
from typing import Dict, Any, Optional, Tuple
import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from src.core.config import get_settings


class OpenMeteoClient:
    """Client for Open-Meteo weather API (free, no key needed)."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.open_meteo_base_url
        self.geolocator = Nominatim(user_agent="agentic-rag-chatbot")

    def geocode_location(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Convert location name to latitude and longitude.

        Args:
            location_name: Name of location (e.g., "New York", "London")

        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            location = self.geolocator.geocode(location_name, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error for '{location_name}': {e}")
            return None

    def get_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        hourly: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Get weather data for location.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            hourly: List of hourly variables to fetch

        Returns:
            Weather data dict
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

        response = requests.get(f"{self.base_url}/forecast", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        daily: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Get historical weather data.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            daily: List of daily variables

        Returns:
            Historical weather data
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
            "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10
        )
        response.raise_for_status()
        return response.json()


class WeatherAnalyzer:
    """Analyzes weather data with safe computations."""

    def __init__(self):
        self.client = OpenMeteoClient()

    def analyze_time_series(
        self, data: Dict[str, Any], variable: str = "temperature_2m"
    ) -> Dict[str, Any]:
        """
        Perform time series analysis on weather data.

        Args:
            data: Weather data from Open-Meteo
            variable: Variable to analyze

        Returns:
            Analysis results
        """
        hourly = data.get("hourly", {})
        values = hourly.get(variable, [])

        if not values:
            return {"error": f"No data found for variable: {variable}"}

        # Safe computations only
        import statistics

        # Filter out None values
        clean_values = [v for v in values if v is not None]

        if not clean_values:
            return {"error": "No valid data points"}

        analysis = {
            "count": len(clean_values),
            "mean": statistics.mean(clean_values),
            "median": statistics.median(clean_values),
            "stdev": statistics.stdev(clean_values) if len(clean_values) > 1 else 0,
            "min": min(clean_values),
            "max": max(clean_values),
            "range": max(clean_values) - min(clean_values),
        }

        # Volatility (coefficient of variation)
        if analysis["mean"] != 0:
            analysis["volatility"] = analysis["stdev"] / abs(analysis["mean"])

        # Missingness check
        missing = len(values) - len(clean_values)
        analysis["missing_count"] = missing
        analysis["missing_percent"] = (missing / len(values)) * 100 if values else 0

        return analysis

    def detect_anomalies(
        self,
        data: Dict[str, Any],
        variable: str = "temperature_2m",
        threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in weather data using Z-score.

        Args:
            data: Weather data
            variable: Variable to check
            threshold: Z-score threshold

        Returns:
            Anomaly detection results
        """
        hourly = data.get("hourly", {})
        values = hourly.get(variable, [])
        times = hourly.get("time", [])

        if not values:
            return {"error": "No data"}

        import statistics

        clean_values = [v for v in values if v is not None]
        if len(clean_values) < 2:
            return {"error": "Insufficient data"}

        mean = statistics.mean(clean_values)
        stdev = statistics.stdev(clean_values)

        if stdev == 0:
            return {"anomalies": [], "message": "No variation in data"}

        anomalies = []
        for i, (value, time) in enumerate(zip(values, times)):
            if value is not None:
                z_score = (value - mean) / stdev
                if abs(z_score) > threshold:
                    anomalies.append(
                        {
                            "time": time,
                            "value": value,
                            "z_score": round(z_score, 2),
                            "direction": "high" if z_score > 0 else "low",
                        }
                    )

        return {
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:10],  # Limit to first 10
            "threshold": threshold,
        }

    def rolling_average(
        self, data: Dict[str, Any], variable: str = "temperature_2m", window: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate rolling average.

        Args:
            data: Weather data
            variable: Variable to analyze
            window: Rolling window size (hours)

        Returns:
            Rolling averages
        """
        hourly = data.get("hourly", {})
        values = hourly.get(variable, [])
        times = hourly.get("time", [])

        if not values:
            return {"error": "No data"}

        # Calculate rolling average
        rolling = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            window_values = [v for v in values[start : i + 1] if v is not None]
            if window_values:
                avg = sum(window_values) / len(window_values)
                rolling.append(
                    {
                        "time": times[i] if i < len(times) else None,
                        "value": values[i],
                        "rolling_avg": round(avg, 2),
                    }
                )

        return {"window_size": window, "rolling_averages": rolling}


class WeatherQueryParser:
    """Parse natural language weather queries."""

    def __init__(self):
        self.metric_keywords = {
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
        Parse natural language weather query.

        Args:
            query: Natural language query (e.g., "What's the humidity in New York last week?")

        Returns:
            Dict with location, metric, time_period, comparison
        """
        import re

        query_lower = query.lower()

        # Extract location - look for "in [Location]" or "for [Location]"
        location = None
        location_patterns = [
            r"in\s+([A-Za-z\s]+?)(?:\s+last|\s+yesterday|\s+today|\s+now|\s+compared|$)",
            r"for\s+([A-Za-z\s]+?)(?:\s+last|\s+yesterday|\s+today|\s+now|\s+compared|$)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                break

        # Extract metric
        metric = "temperature_2m"  # default
        for keyword, api_metric in self.metric_keywords.items():
            if keyword in query_lower:
                metric = api_metric
                break

        # Extract time period
        time_period = "current"
        if "last week" in query_lower or "past week" in query_lower:
            time_period = "last_week"
        elif "yesterday" in query_lower:
            time_period = "yesterday"
        elif "last month" in query_lower or "past month" in query_lower:
            time_period = "last_month"

        # Check for comparison
        comparison = (
            "compared" in query_lower or "vs" in query_lower or "versus" in query_lower
        )

        return {
            "location": location,
            "metric": metric,
            "time_period": time_period,
            "comparison": comparison,
            "original_query": query,
        }

    def get_date_range(self, time_period: str) -> tuple:
        """
        Convert time period to date range.

        Args:
            time_period: Time period string (e.g., "last_week")

        Returns:
            Tuple of (start_date, end_date) as strings (YYYY-MM-DD)
        """
        today = datetime.now()
        end_date = today.strftime("%Y-%m-%d")

        if time_period == "yesterday":
            start = today - timedelta(days=1)
            start_date = start.strftime("%Y-%m-%d")
        elif time_period == "last_week":
            start = today - timedelta(days=7)
            start_date = start.strftime("%Y-%m-%d")
        elif time_period == "last_month":
            start = today - timedelta(days=30)
            start_date = start.strftime("%Y-%m-%d")
        else:  # current
            start_date = end_date

        return (start_date, end_date)
