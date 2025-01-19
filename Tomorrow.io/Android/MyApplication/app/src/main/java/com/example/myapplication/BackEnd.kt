package com.example.myapplication

//import org.bson.Document
//import org.bson.types.ObjectId
//import org.litote.kmongo.coroutine.*
//import org.litote.kmongo.eq
//import org.litote.kmongo.reactivestreams.KMongo

import com.example.myapplication.models.WeatherData
import com.example.myapplication.models.DailyWeather
import com.example.myapplication.models.Location
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.roundToInt


suspend fun IPInfoLocation(): Location? {
    val url = "https://ipinfo.io?token=dc9d2d418e567f"

    return withContext(Dispatchers.IO) {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.requestMethod = "GET"

        if (connection.responseCode == HttpURLConnection.HTTP_OK) {
            val response = connection.inputStream.bufferedReader().use { it.readText() }
            val jsonResponse = JSONObject(response)

            val loc = jsonResponse.getString("loc").split(",").map { it.toDouble() }
            val city = jsonResponse.getString("city")
            val state = jsonResponse.getString("region")

//            Location(
//                city = city, state = state, latitude = loc[0], longitude = loc[1]
//            )
            Location(
                city = "Los Angeles", state = "California", latitude = 34.0195, longitude = -118.4914
            )
        } else {
            null
        }
    }
}

// 修改后的 getWeatherData，作为挂起函数
suspend fun getWeatherData(location: Location?, appKey: String): WeatherData {
    val fields = listOf(
        "temperature", "humidity", "windSpeed", "visibility", "pressureSeaLevel",
        "precipitationIntensity", "cloudCover", "weatherCode", "temperatureMin", "temperatureMax", "uvIndex"
    ).joinToString(",")

    val apiUrl =
        "https://api.tomorrow.io/v4/timelines?location=${location?.latitude},${location?.longitude}&fields=$fields&timesteps=1d&units=metric&timezone=auto&apikey=$appKey"

    println(apiUrl)
    return withContext(Dispatchers.IO) {
        try {
            // 发起 HTTP 请求
            val connection = URL(apiUrl).openConnection() as HttpURLConnection
            connection.requestMethod = "GET"

            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                val jsonResponse = JSONObject(response)
                // 解析响应并返回 WeatherData
                parseWeatherData(jsonResponse, location)
            } else {
                throw Exception("Failed to fetch weather data: HTTP ${connection.responseCode}")
            }
        } catch (e: Exception) {
            throw Exception("Error fetching weather data: ${e.message}")
        }
    }
}

private fun parseWeatherData(response: JSONObject, location: Location?): WeatherData {
    val data = response.getJSONObject("data")
    val timelines = data.getJSONArray("timelines")

    // 提取 1d 数据
    val dailyTimeline = timelines.getJSONObject(0).getJSONArray("intervals")
    val dailyWeatherList = mutableListOf<DailyWeather>()

    // 定义时间格式解析器
    val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.getDefault()) // 解析带时区的时间

    for (i in 0 until dailyTimeline.length()) {
        val interval = dailyTimeline.getJSONObject(i)
        val values = interval.getJSONObject("values")

        // 获取天气代码
        val weatherCode = values.getInt("weatherCode")

        // 解析时间为 Unix 时间戳
        val rawTime = interval.getString("startTime") // 例如 "2024-11-29T06:00:00-08:00"
        val timestamp = try {
            // 将字符串时间解析为 Date 对象并获取 Unix 时间戳（秒级）
            val date = sdf.parse(rawTime)
            (date?.time ?: 0L) / 1000 // 转换为秒
        } catch (e: Exception) {
            e.printStackTrace()
            0L // 如果解析失败，返回 0
        }

        // 转换 temperatureMin 和 temperatureMax 为华氏温度
        val temperatureLowFahrenheit = ((values.getDouble("temperatureMin") * 9 / 5 + 32).roundToInt()).toString()
        val temperatureHighFahrenheit = ((values.getDouble("temperatureMax") * 9 / 5 + 32).roundToInt()).toString()

        // 构建 DailyWeather 对象
        val dailyWeather = DailyWeather(
            time = timestamp.toString(), // 使用 Unix 时间戳字符串
            weather_name = WeatherMapping.WeatherNameMapping[weatherCode] ?: "Unknown",
            weather_image = WeatherMapping.WeatherImageMapping[weatherCode] ?: R.drawable.ic_launcher_background,
            temperatureLow = temperatureLowFahrenheit,
            temperatureHigh = temperatureHighFahrenheit
        )

        dailyWeatherList.add(dailyWeather)
    }

    // 提取当前天气数据
    val currentValues = dailyTimeline.getJSONObject(0).getJSONObject("values")
    val currentWeatherCode = currentValues.getInt("weatherCode")

    // 构建 WeatherData 对象
    return WeatherData(
        temperature = (currentValues.getDouble("temperature") * 9 / 5 + 32).roundToInt().toFloat(), // 华氏温度，取整后转为 float
        weather_name = WeatherMapping.WeatherNameMapping[currentWeatherCode] ?: "Unknown", // 天气名称映射
        weather_image = WeatherMapping.WeatherImageMapping[currentWeatherCode] ?: R.drawable.ic_launcher_background, // 天气图片映射
        city = location?.city ?: "Unknown", // 城市信息
        state = location?.state ?: "Unknown",
        humidity = currentValues.getDouble("humidity").toFloat(), // 湿度
        windSpeed = currentValues.getDouble("windSpeed").toFloat(), // 风速
        visibility = currentValues.getDouble("visibility").toFloat(), // 能见度
        pressure = String.format("%.1f", currentValues.getDouble("pressureSeaLevel") * 0.02953).toFloat(), // 气压转换为 inHg，保留一位小数
        precipitation = currentValues.getDouble("precipitationIntensity").toFloat(), // 降水量
        cloud_cover = currentValues.getDouble("cloudCover").toFloat(), // 云量
        ozone = currentValues.optDouble("uvIndex", 0.0).toFloat(), // 紫外线指数
        daily = dailyWeatherList // 每日天气数据
    )
}


suspend fun Search_Location(Add_String: String, googleApiKey: String): Location? {
    return withContext(Dispatchers.IO) {
        try {
            // 构建请求 URL
            val url = URL("https://maps.googleapis.com/maps/api/geocode/json?address=${Add_String}&key=${googleApiKey}")

            // 打开连接
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"

            // 检查响应码
            if (connection.responseCode == HttpURLConnection.HTTP_OK) {
                // 读取响应数据
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                val jsonResponse = JSONObject(response)

                // 解析 JSON 数据
                val results = jsonResponse.getJSONArray("results")
                if (results.length() > 0) {
                    val firstResult = results.getJSONObject(0)
                    val addressComponents = firstResult.getJSONArray("address_components")
                    val geometry = firstResult.getJSONObject("geometry").getJSONObject("location")

                    // 提取城市和州信息
                    var city = ""
                    var state = ""
                    for (i in 0 until addressComponents.length()) {
                        val component = addressComponents.getJSONObject(i)
                        val types = component.getJSONArray("types")
                        when {
                            types.toString().contains("locality") -> city = component.getString("long_name")
                            types.toString().contains("administrative_area_level_1") -> state = component.getString("long_name")
                        }
                    }

                    // 提取经纬度
                    val latitude = geometry.getDouble("lat")
                    val longitude = geometry.getDouble("lng")

                    // 返回 Location 数据
                    Location(
                        city = city,
                        state = state,
                        latitude = latitude,
                        longitude = longitude
                    )
                } else {
                    null // 如果没有结果，返回 null
                }
            } else {
                null // 请求失败，返回 null
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null // 出现异常时返回 null
        }
    }
}


//class BackendHelper(
//    context: Context, // 需要 Android 上下文来初始化 Volley
//    private val collection: CoroutineCollection<Document>, // 注入 MongoDB 集合
//    private val appKey: String, // Tomorrow.io 的 API 密钥
//    private val googleApiKey: String // Google API 密钥
//) {
//    private val requestQueue: RequestQueue = Volley.newRequestQueue(context)
//
//    /** 获取 MongoDB 数据 */
//    suspend fun getDataFromMongo(): List<Document> {
//        return try {
//            collection.find().toList()
//        } catch (e: Exception) {
//            println("Error fetching data from MongoDB: ${e.message}")
//            throw e
//        }
//    }
//
//    /** 插入数据到 MongoDB */
//    suspend fun insertDataToMongo(newData: Document): String {
//        return try {
//            val result = collection.insertOne(newData)
//            result.insertedId?.toString() ?: throw Exception("Failed to insert data")
//        } catch (e: Exception) {
//            println("Error inserting data into MongoDB: ${e.message}")
//            throw e
//        }
//    }
//
//    /** 删除 MongoDB 数据 */
//    suspend fun deleteDataFromMongo(documentId: String): Boolean {
//        return try {
//            val result = collection.deleteOne(eq("_id", ObjectId(documentId)))
//            result.deletedCount == 1L
//        } catch (e: Exception) {
//            println("Error deleting data from MongoDB: ${e.message}")
//            throw e
//        }
//    }
//
//    /** 获取天气数据 */

//
//
//// Functions to process the data
//fun dataProcess1D(json1D: String): Map<String, Map<String, String>> {
//    // 创建 Json 对象，启用 ignoreUnknownKeys
//
//    val json = Json {
//        ignoreUnknownKeys = true // 忽略未知字段
//    }
//
//    val resultData = mutableMapOf<String, Map<String, String>>()
//    try {
//        val jsonData = json.decodeFromString<Json1D>(json1D)
//        val weatherData = jsonData.data.timelines[0].intervals
//
//        weatherData.forEachIndexed { idx, interval ->
//            val inter = interval.values
//            val curWeatherCode = inter.weatherCode
//
//            // 格式化时间
//            val formattedTime = try {
//                val startTime = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssXXX", Locale.US).parse(interval.startTime)
//                startTime?.let {
//                    val formatter = SimpleDateFormat("EEEE, MMM dd, yyyy", Locale.US)
//                    formatter.format(it)
//                } ?: "Unknown Date"
//            } catch (e: Exception) {
//                "Unknown Date"
//            }
//
//            // 数据行
//            val row = mapOf(
//                "Time" to formattedTime,
//                "humidity" to "${inter.humidity?.toString() ?: "N/A"}%",
//                "precipitationProbability" to "${inter.precipitationProbability?.toString() ?: "N/A"}%",
//                "precipitationType" to (inter.precipitationType?.toString() ?: "N/A"),
//                "sunriseTime" to (inter.sunriseTime ?: "N/A"),
//                "sunsetTime" to (inter.sunsetTime ?: "N/A"),
//                "temperatureApparent" to "${inter.temperatureApparent?.toString() ?: "N/A"}°",
//                "temperatureMax" to "${inter.temperatureMax?.toString() ?: "N/A"}°",
//                "temperatureMin" to "${inter.temperatureMin?.toString() ?: "N/A"}°",
//                "visibility" to "${inter.visibility?.toString() ?: "N/A"}mi",
//                "Image" to "/${WeatherMapping.WeatherImageMapping[curWeatherCode] ?: "unknown.png"}",
//                "Weather" to (WeatherMapping.WeatherNameMapping[curWeatherCode] ?: "Unknown"),
//                "windSpeed" to "${inter.windSpeed?.toString() ?: "N/A"}mph",
//                "cloudcover" to "${inter.cloudCover?.toString() ?: "N/A"}%"
//            )
//
//            val dayKey = "day${idx + 1}"
//            resultData[dayKey] = row
//        }
//    } catch (e: Exception) {
//        println("Error processing data: ${e.message}")
//    }
//
//    return resultData
//}

