package com.example.myapplication.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.myapplication.models.Location
import com.example.myapplication.models.WeatherData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

//class WeatherViewModel(
//    initialWeatherData: WeatherData? = null,
//    initialLocation: Location? = null
//) : ViewModel() {
//
//    private val _weather = MutableStateFlow(initialWeatherData)
//    val weather: StateFlow<WeatherData?> = _weather
//
//    private val _location = MutableStateFlow(initialLocation)
//    val location: StateFlow<Location?> = _location
//
//    fun updateWeatherData(weatherData: WeatherData) {
//        _weather.value = weatherData
//    }
//
//    fun updateLocation(location: Location) {
//        _location.value = location
//    }
//}