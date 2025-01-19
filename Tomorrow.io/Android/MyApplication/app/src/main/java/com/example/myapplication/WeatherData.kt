package com.example.myapplication.models

data class WeatherDocument(
    val locations: List<Location> = emptyList(),
    val weathers: List<WeatherData> = emptyList()
)

data class WeatherData(
    val temperature: Float,
    val weather_name: String,
    val weather_image: Int,
    val city: String,
    val state: String,
    val humidity: Float,
    val windSpeed: Float,
    val visibility: Float,
    val pressure: Float,
    val precipitation: Float,
    val cloud_cover: Float,
    val ozone: Float,
    val daily: List<DailyWeather>
)

data class DailyWeather(
    val time: String,
    val weather_name: String,
    val weather_image: Int,
    val temperatureLow: String,
    val temperatureHigh: String
)

data class Location(
    val city: String,
    val state: String,
    val latitude: Double,
    val longitude: Double
)