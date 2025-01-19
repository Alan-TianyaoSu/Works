package com.example.myapplication.viewmodels

import android.provider.DocumentsContract.Document
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.myapplication.IPInfoLocation
import com.example.myapplication.Search_Location
import com.example.myapplication.appKey
import com.example.myapplication.getWeatherData
import com.example.myapplication.models.DailyWeather
import com.example.myapplication.models.WeatherData
import com.example.myapplication.models.Location
import com.example.myapplication.models.WeatherDocument
import com.mongodb.reactivestreams.client.MongoClients
import com.mongodb.reactivestreams.client.MongoCollection
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

//import com.mongodb.client.MongoClients
//import com.mongodb.client.MongoCollection
//import org.bson.Document

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.litote.kmongo.coroutine.coroutine
import org.litote.kmongo.reactivestreams.KMongo


import kotlinx.coroutines.withContext
import org.litote.kmongo.coroutine.coroutine

class WeatherViewModel : ViewModel() {

    private val _location = MutableStateFlow<Location?>(null) // 当前单个位置信息
    val location: StateFlow<Location?> = _location

    private val _weather = MutableStateFlow<WeatherData?>(null) // 当前单个天气数据
    val weather: StateFlow<WeatherData?> = _weather

    private val _locations = MutableStateFlow<List<Location>>(emptyList())
    val locations: StateFlow<List<Location>> = _locations

    private val _weathers = MutableStateFlow<List<WeatherData>>(emptyList())
    val weathers: StateFlow<List<WeatherData>> = _weathers

    private val _isLoading = MutableStateFlow(true) // 加载状态
    val isLoading: StateFlow<Boolean> = _isLoading

    private val _isSearch = MutableStateFlow(false) // 是否通过搜索获取
    val isSearch: StateFlow<Boolean> = _isSearch

    // 新增 city 和 state 的 StateFlow
    private val _city = MutableStateFlow<String?>(null) // 当前搜索到的城市
    val city: StateFlow<String?> = _city

    private val _state = MutableStateFlow<String?>(null) // 当前搜索到的州/省份
    val state: StateFlow<String?> = _state

    private val _pageNumber = MutableStateFlow(0) // 当前页面编号
    val pageNumber: StateFlow<Int> = _pageNumber

    init {
        viewModelScope.launch {
            try {
                // (1) 获取初始位置信息
                val initialLocation = IPInfoLocation()
                // (2) 获取初始天气数据
                val initialWeather = getWeatherData(location = initialLocation, appKey = appKey)
                println("initialWeather")
                println(initialWeather)
                // (3) 设置当前的单个位置和天气
                _location.value = initialLocation
                _weather.value = initialWeather

                // (4) 将初始数据添加到列表中
                initialLocation?.let {
                    _locations.value = _locations.value + it
                }
                _weathers.value = _weathers.value + initialWeather

                // (5) 从 MongoDB 同步数据
//                Synchron_from_DB()
                println("Synchron complete!~")
            } catch (e: Exception) {
                println("Error loading data: ${e.message}")
            } finally {
                _isLoading.value = false // 数据加载完成
            }
        }
    }

    // 搜索新位置并更新当前位置和天气，同时添加到列表中
    fun searchLocation(query: String, googleApiKey: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _isSearch.value = true // 标记为搜索状态
            try {
                // 将输入的 "city, state" 转换为 "city+state"
                val formattedQuery = query.replace(", ", "+").replace(" ", "+")
                val parts = query.split(", ")

                if (parts.size == 2) {
                    val city = parts[0]
                    val state = parts[1]

                    // 更新 city 和 state 的值
                    _city.value = city
                    _state.value = state
                } else {
                    println("Invalid query format. Expected 'city, state'.")
                    return@launch
                }

                // 调用 Search_Location 挂起函数
                val result = Search_Location(formattedQuery, googleApiKey)

                if (result != null) {
                    // 获取新的天气数据
                    val newWeather = getWeatherData(location = result, appKey = appKey)

                    // 更新当前的单个位置和天气
                    _location.value = result
                    _weather.value = newWeather

                    // 动态添加新数据到列表中
//                     _locations.value = _locations.value + result
//                     _weathers.value = _weathers.value + newWeather
                } else {
                    println("No location found for query: $query")
                }
            } catch (e: Exception) {
                println("Error searching location: ${e.message}")
            } finally {
                _isLoading.value = false
            }
        }
    }

    // 重置搜索标记为 false
    fun resetSearchFlag() {
        _isSearch.value = false
    }

    fun AddToFavorite() {
        if (_location.value != null && _weather.value != null) {
            // 动态将当前位置添加到收藏列表中
            _locations.value = _locations.value.orEmpty() + _location.value!!
            _weathers.value = _weathers.value.orEmpty() + _weather.value!!
        }
//        Synchron_to_DB()
    }

    fun RemoveFromFavorite() {
        val locationExists = _locations.value.orEmpty().any { it.city == city.value }
        val weatherExists = _weathers.value.orEmpty().any { it.city == city.value }

        if (!locationExists || !weatherExists) {
            println("City: $city does not exist in favorites")
            return
        }

        val updatedLocations = _locations.value.orEmpty().filterNot { it.city == city.value }
        val updatedWeathers = _weathers.value.orEmpty().filterNot { it.city == city.value }

        _locations.value = updatedLocations
        _weathers.value = updatedWeathers

        println("Removed city: $city from favorites")
    }

    fun removeCity(index: Int) {
        viewModelScope.launch {
            // 获取当前的列表
            val currentLocations = _locations.value.orEmpty()
            val currentWeathers = _weathers.value.orEmpty()

            // 检查下标是否有效
            if (index < 0 || index >= currentLocations.size || index >= currentWeathers.size) {
                println("Index $index is out of bounds. Cannot remove.")
                return@launch
            }

            // 移除指定下标的城市
            val updatedLocations = currentLocations.toMutableList().apply { removeAt(index) }
            val updatedWeathers = currentWeathers.toMutableList().apply { removeAt(index) }

            // 更新 StateFlow
            _locations.value = updatedLocations
            _weathers.value = updatedWeathers

            // 如果被移除的城市是当前选中的城市，清空当前城市状态（可选）
            if (_city.value == currentLocations[index].city) {
                _city.value = null
            }

            println("Removed city at index $index: ${currentLocations[index].city}")
        }
    }

    fun updatePageNumber(newPageNumber: Int) {
        _pageNumber.value = newPageNumber
    }

}