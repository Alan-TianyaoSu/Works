package com.example.myapplication

import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import com.example.myapplication.viewmodels.WeatherViewModel
//import com.example.myapplication.viewmodels.WeatherViewModelFactory
import com.example.myapplication.models.DailyWeather

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.FloatingActionButton
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.zIndex
import com.android.volley.Request
import com.android.volley.toolbox.StringRequest
import com.android.volley.toolbox.Volley
import com.google.accompanist.pager.ExperimentalPagerApi
import com.google.accompanist.pager.HorizontalPager
import com.google.accompanist.pager.HorizontalPagerIndicator
import com.google.accompanist.pager.rememberPagerState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONArray
import org.json.JSONObject

import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException


@OptIn(ExperimentalMaterial3Api::class, ExperimentalPagerApi::class)
@Composable
fun HomeScreen(
    weatherViewModel: WeatherViewModel,
    onCardClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var isSearchVisible by remember { mutableStateOf(false) }
    var searchCompleted by remember { mutableStateOf(false) }
    var searchText by remember { mutableStateOf("") }
    var suggestions by remember { mutableStateOf<List<String>>(listOf()) }
    val weatherList by weatherViewModel.weathers.collectAsState()
    val LocationList by weatherViewModel.locations.collectAsState()
    val searchFlag by weatherViewModel.isSearch.collectAsState()
    var expanded by remember { mutableStateOf(false) }
    val weatherData = weatherViewModel.weather.collectAsState().value
    val location = weatherViewModel.location.collectAsState().value

    val city by weatherViewModel.city.collectAsState()
    val state by weatherViewModel.state.collectAsState()

    // 定义一个上下文
    val context = LocalContext.current

    Scaffold(
        containerColor = Color(0xFF000000),
        topBar = {

            CenterAlignedTopAppBar(
                modifier = Modifier.fillMaxWidth(),
                title = {

                    if (isSearchVisible) {
                            Column {
                                // 搜索栏
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .background(Color.Black) // 搜索栏背景色
                                        .padding(horizontal = 8.dp, vertical = 8.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    IconButton(onClick = {
                                        isSearchVisible = false
                                        searchCompleted = false
                                        searchText = ""
                                        suggestions = emptyList() // 清空搜索建议
                                    }) {
                                        Icon(
                                            imageVector = Icons.Default.ArrowBack,
                                            contentDescription = "Back",
                                            tint = Color.White
                                        )
                                    }

                                    TextField(
                                        value = searchText,
                                        onValueChange = { query ->
                                            searchText = query
                                            expanded = query.isNotEmpty()
                                            if (query.isNotEmpty()) {
                                                CoroutineScope(Dispatchers.Main).launch {
                                                    suggestions = fetchSuggestionsFromApi(query, context)
                                                }
                                            } else {
                                                suggestions = emptyList()
                                            }
                                        },
                                        placeholder = { Text("Search...") },
                                        singleLine = true,
                                        colors = TextFieldDefaults.textFieldColors(
                                            focusedTextColor = Color.White,
                                            unfocusedTextColor = Color.White,
                                            cursorColor = Color.White,
                                            containerColor = Color.Transparent,
                                            focusedPlaceholderColor = Color.Gray,
                                            unfocusedPlaceholderColor = Color.Gray,
                                            focusedIndicatorColor = Color.Transparent,
                                            unfocusedIndicatorColor = Color.Transparent
                                        ),
                                        modifier = Modifier
                                            .weight(1f)
                                    )

                                    // 在搜索完成后隐藏清除按钮
                                    if (!searchCompleted) {
                                        IconButton(onClick = { searchText = "" }) {
                                            Icon(
                                                imageVector = Icons.Default.Close,
                                                contentDescription = "Clear",
                                                tint = Color.White
                                            )
                                        }
                                    }
                                }
                            }
                        }
                       else {
                            // 标题栏
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(0.dp)
                                    .background(if (searchFlag) Color(0xFF262626) else Color.Black),
//                                    .fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically
                            ) {


                                // 搜索完成后显示灰色的不可点击标题
                                if (searchFlag) {

                                    IconButton(onClick = {
                                        isSearchVisible = false
                                        searchCompleted = false
                                        searchText = ""
                                        suggestions = emptyList() // 清空搜索建议
                                        weatherViewModel.resetSearchFlag()
                                    }) {
                                        Icon(
                                            imageVector = Icons.Default.ArrowBack,
                                            contentDescription = "Back",
                                            tint = Color.White
                                        )
                                    }

                                    Text(
                                        text = "${city}, ${state}", // 显示搜索的城市和州
                                        color = Color.White,
                                        modifier = Modifier
                                            .weight(1f)
                                            .padding(start = 8.dp)
                                    )

                                } else {
                                    Text(
                                        text = "WeatherApp",
                                        color = Color.White,
                                        modifier = Modifier
                                            .weight(1f)
                                            .padding(start = 8.dp)
                                    )

                                    // 返回按钮始终可用
                                    IconButton(
                                        onClick = { isSearchVisible = true } // 点击后显示搜索框
                                    ) {
                                        Icon(
                                            painter = painterResource(id = com.example.myapplication.R.drawable.map_search),
                                            contentDescription = "Search",
                                            tint = Color.White
                                        )
                                    }
                                }
                            }
                        }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = Color(0xFF000000) // 顶部栏背景颜色
                )
            )
        },
        content = { padding ->
            Column( // 外层使用 Column 包裹整个布局
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f) // 让 Box 占据剩余空间
                        .fillMaxWidth()
                ) {
                    println(searchFlag)

                    if (searchFlag) {
                        weatherData?.let {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color(0xFF000000))
                            ) {
                                // 主内容区域
                                Column(
                                    modifier = Modifier
                                        .fillMaxSize()
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color(0xFF000000))
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .fillMaxSize()
                                                .background(Color(0xFF111111))
                                                .padding(start = 6.dp, end = 6.dp)
                                        ) {
                                            Column(
                                                modifier = Modifier.padding(bottom = 20.dp)
                                            ) {
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .background(Color(0xFF111111))
                                                        .padding(start = 16.dp, end = 16.dp)
                                                        .height(70.dp)
                                                ) {
                                                    Text(
                                                        text = "Search Result", // 搜索结果标题
                                                        color = Color(0xFFc0c0c0),
                                                        style = androidx.compose.ui.text.TextStyle(
                                                            fontSize = 24.sp
                                                        ),
                                                        modifier = Modifier.align(Alignment.CenterStart)
                                                    )
                                                }

                                                // 当前天气卡片
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .background(Color(0xFF2b2b2b))
                                                        .padding(vertical = 16.dp)
                                                ) {
                                                    CurrentWeatherCard(
                                                        temperature = it.temperature,
                                                        summary = it.weather_name,
                                                        city = "${it.city}, ${it.state}",
                                                        onClick = onCardClick,
                                                        image_id = it.weather_image
                                                    )
                                                }

                                                // 天气详情卡片
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .background(Color(0xFF2b2b2b))
                                                ) {
                                                    WeatherDetailsCard(
                                                        humidity = it.humidity,
                                                        windSpeed = it.windSpeed,
                                                        visibility = it.visibility,
                                                        pressure = it.pressure
                                                    )
                                                }

                                                // 每周天气预报卡片
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .background(Color(0xFF2b2b2b))
                                                        .padding(vertical = 16.dp)
                                                ) {
                                                    WeeklyForecastCard(dailyWeather = it.daily)
                                                }
                                            }
                                        }
                                    }
                                }

                                // 右下角 FloatingActionButton
                                val isFavorite = remember { mutableStateOf(false) }
                                val isExisting =  LocationList.any { it.city == city }
                                if (isExisting) {
                                    isFavorite.value = true
                                } else {
                                    isFavorite.value = false
                                }

                                val context = LocalContext.current

                                FloatingActionButton(
                                    onClick = {
                                        // 检查当前城市是否已在收藏列表中
                                        val isExisting =  LocationList.any { it.city == city }

                                        if (isExisting) {
                                            // 如果存在，移除该城市
                                            weatherViewModel.RemoveFromFavorite()
                                            isFavorite.value = false
                                        } else {
                                            // 如果不存在，添加到收藏
                                            weatherViewModel.AddToFavorite()
                                            isFavorite.value = true
                                        }

                                        // 根据操作生成对应的提示信息
                                        val message = if (isFavorite.value) {
                                            "$city was added to favorites"
                                        } else {
                                            "$city was removed from favorites"
                                        }

                                        // 自定义 Toast
                                        val toast = Toast(context)
                                        val layout = LinearLayout(context).apply {
                                            orientation = LinearLayout.HORIZONTAL
                                            setBackgroundResource(android.R.drawable.toast_frame) // 设置默认背景
                                            setPadding(30, 24, 30, 24)

                                            // 添加图标
                                            val icon = ImageView(context).apply {
                                                setImageResource(R.drawable.weather_partly_snowy_rainy) // 设置自定义图标
                                                setColorFilter(android.graphics.Color.WHITE) // 设置图案颜色为白色

                                                // 设置背景为圆形，并填充颜色
                                                background = GradientDrawable().apply {
                                                    shape = GradientDrawable.OVAL // 设置为圆形
                                                    setColor(android.graphics.Color.parseColor("#3C71DB")) // 设置底色为 #3C71DB
                                                }

                                                // 设置内边距，让图案变小并居中
                                                setPadding(12, 12, 12, 12) // 调整内边距，缩小图案的显示区域

                                                // 设置容器大小，背景尺寸保持不变
                                                layoutParams = LinearLayout.LayoutParams(64, 64).apply { // 背景大小
                                                    setMargins(16, 0, 16, 0) // 设置边距
                                                    gravity = Gravity.CENTER // 确保图案水平和垂直居中
                                                }

                                                // 确保图案在背景中居中显示
                                                scaleType = ImageView.ScaleType.CENTER_INSIDE
                                            }
                                            addView(icon) // 将图标添加到布局中

                                            // 添加文字
                                            val textView = TextView(context).apply {
                                                text = message
                                                setTextColor(android.graphics.Color.BLACK) // 设置文字颜色
                                                textSize = 14f // 设置文字大小
                                                gravity = Gravity.CENTER
                                                setPadding(16, 0, 0, 0) // 设置文字与图标之间的间距
                                            }
                                            addView(textView) // 将文字添加到布局中
                                        }

                                        toast.view = layout
                                        toast.duration = Toast.LENGTH_SHORT
                                        toast.show()
                                    },
                                    containerColor = Color.White, // 按钮背景色
                                    shape = CircleShape, // 确保按钮是圆形
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd) // 对齐到右下角
                                        .padding(end = 16.dp, bottom = 120.dp) // 设置按钮边距
                                ) {
                                    // 根据收藏状态切换图标
                                    Icon(
                                        painter = painterResource(
                                            id = if (isFavorite.value) R.drawable.rem_fav else R.drawable.add_fav
                                        ),
                                        contentDescription = if (isFavorite.value) "Remove from favorites" else "Add to favorites",
                                        tint = Color.Black // 图标颜色
                                    )
                                }
                            }
                        }
                    } else {


                        val pagerState = rememberPagerState() // 用于追踪当前页面状态
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color(0xFF111111))
                                .padding(start = 6.dp, end = 6.dp)
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color(0xFF111111))
                            ) {
                                // 顶部的小点点指示器
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(70.dp)
                                        .background(Color(0xFF111111))
                                        .padding(vertical = 16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    HorizontalPagerIndicator(
                                        pagerState = pagerState, // 绑定 HorizontalPager 的状态
                                        activeColor = Color.White, // 当前页面小点点颜色
                                        inactiveColor = Color.Gray, // 非当前页面小点点颜色
                                        modifier = Modifier.padding(8.dp) // 给指示器一些内边距
                                    )
                                }

                                LaunchedEffect(pagerState) {
                                    snapshotFlow { pagerState.currentPage } // 观察 currentPage 的变化
                                        .collect { page ->
                                            weatherViewModel.updatePageNumber(page) // 同步页面编号到 ViewModel
                                        }
                                }

                                // HorizontalPager 实现滑动效果
                                HorizontalPager(
                                    count = weatherList.size, // weatherList 是你的 WeatherData 列表
                                    state = pagerState,
                                    modifier = Modifier.fillMaxSize()
                                ) { page ->
                                    val weatherData = weatherList[page]



                                    Column(
                                        modifier = Modifier
                                            .fillMaxSize()
                                    ) {
                                        // 当前天气卡片
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .background(Color(0xFF2b2b2b))
                                                .padding(vertical = 16.dp)
                                        ) {
                                            CurrentWeatherCard(
                                                temperature = weatherData.temperature,
                                                summary = weatherData.weather_name,
                                                city = "${weatherData.city}, ${weatherData.state}",
                                                onClick = onCardClick,
                                                image_id = weatherData.weather_image
                                            )
                                        }

                                        // 天气详情卡片
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .background(Color(0xFF2b2b2b))
                                        ) {
                                            WeatherDetailsCard(
                                                humidity = weatherData.humidity,
                                                windSpeed = weatherData.windSpeed,
                                                visibility = weatherData.visibility,
                                                pressure = weatherData.pressure
                                            )
                                        }

                                        // 每周天气预报卡片
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .background(Color(0xFF2b2b2b))
                                                .padding(vertical = 16.dp)
                                        ) {
                                            WeeklyForecastCard(dailyWeather = weatherData.daily)
                                        }
                                    }
                                }
                            }

                            // 添加按钮：只在不是第一页时显示
                            if (pagerState.currentPage != 0) {

                                FloatingActionButton(
                                    onClick = {
                                        // 检查当前城市是否已在收藏列表中
                                        val isExisting = LocationList.any { it.city == weatherList[pagerState.currentPage].city }

                                        if (isExisting) {
                                            // 如果存在，移除该城市
                                            weatherViewModel.removeCity(pagerState.currentPage)

                                        } else {
                                            // 如果不存在，添加到收藏
                                            weatherViewModel.AddToFavorite()
                                        }

                                        // 根据操作生成对应的提示信息
                                        val message = "${weatherList[pagerState.currentPage].city} was removed from favorites"


                                        // 自定义 Toast
                                        val toast = Toast(context)
                                        val layout = LinearLayout(context).apply {
                                            orientation = LinearLayout.HORIZONTAL
                                            setBackgroundResource(android.R.drawable.toast_frame) // 设置默认背景
                                            setPadding(30, 24, 30, 24)

                                            // 添加图标
                                            val icon = ImageView(context).apply {
                                                setImageResource(R.drawable.weather_partly_snowy_rainy) // 设置自定义图标
                                                setColorFilter(android.graphics.Color.WHITE) // 设置图案颜色为白色

                                                // 设置背景为圆形，并填充颜色
                                                background = GradientDrawable().apply {
                                                    shape = GradientDrawable.OVAL // 设置为圆形
                                                    setColor(android.graphics.Color.parseColor("#3C71DB")) // 设置底色为 #3C71DB
                                                }

                                                // 设置内边距，让图案变小并居中
                                                setPadding(12, 12, 12, 12) // 调整内边距，缩小图案的显示区域

                                                // 设置容器大小，背景尺寸保持不变
                                                layoutParams = LinearLayout.LayoutParams(64, 64).apply { // 背景大小
                                                    setMargins(16, 0, 16, 0) // 设置边距
                                                    gravity = Gravity.CENTER // 确保图案水平和垂直居中
                                                }

                                                // 确保图案在背景中居中显示
                                                scaleType = ImageView.ScaleType.CENTER_INSIDE
                                            }
                                            addView(icon) // 将图标添加到布局中

                                            // 添加文字
                                            val textView = TextView(context).apply {
                                                text = message
                                                setTextColor(android.graphics.Color.BLACK) // 设置文字颜色
                                                textSize = 14f // 设置文字大小
                                                gravity = Gravity.CENTER
                                                setPadding(16, 0, 0, 0) // 设置文字与图标之间的间距
                                            }
                                            addView(textView) // 将文字添加到布局中
                                        }

                                        toast.view = layout
                                        toast.duration = Toast.LENGTH_SHORT
                                        toast.show()
                                    },
                                    containerColor = Color.White, // 按钮背景色
                                    shape = CircleShape, // 确保按钮是圆形
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd) // 对齐到右下角
                                        .padding(end = 16.dp, bottom = 120.dp) // 设置按钮边距
                                ) {
                                    // 根据收藏状态切换图标
                                    Icon(
                                        painter = painterResource(
                                            id = R.drawable.rem_fav
                                        ),
                                        contentDescription = "Remove from favorites",
                                        tint = Color.Black // 图标颜色
                                    )
                                }
                            }
                        }
                    }

                    // 下拉列表部分
                    if (suggestions.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .align(Alignment.TopCenter)
                                .width(300.dp)
                                .heightIn(max = 200.dp)
                                .background(Color(0xFF101010))
                                .padding(horizontal = 8.dp)
                                .zIndex(1f)
                        ) {
                            LazyColumn {
                                items(suggestions) { suggestion ->
                                    Text(
                                        text = suggestion,
                                        color = Color.White,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 12.dp)
                                            .clickable {
                                                searchText = suggestion
                                                searchCompleted = true
                                                isSearchVisible = false
                                                println(searchCompleted)
                                                suggestions = emptyList()
                                                // 调用 ViewModel 的 searchLocation 方法
                                                weatherViewModel.searchLocation(suggestion, googleApiKey)
                                            },
                                        style = androidx.compose.ui.text.TextStyle(
                                            fontSize = 18.sp
                                        )
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    )
}

// API 调用函数
suspend fun fetchSuggestionsFromApi(query: String, context: Context): List<String> {
    val googleApiKey = "AIzaSyBOI4I20qLdWyWRU3ha2r_SErsdCqTkCRg" // 替换为你的 Google API Key
    val url = "https://assignment3-784518.wl.r.appspot.com/Get_autocomplete?input=$query"

    return suspendCancellableCoroutine { continuation ->
        val requestQueue = Volley.newRequestQueue(context)

        val request = StringRequest(
            Request.Method.GET,
            url,
            { response ->
                try {
                    val suggestions = mutableListOf<String>()
                    val jsonObject = JSONObject(response)
                    val predictionsArray: JSONArray = jsonObject.getJSONArray("predictions")
                    for (i in 0 until predictionsArray.length()) {
                        val prediction = predictionsArray.getJSONObject(i)
                        val termsArray = prediction.getJSONArray("terms")

                        // 获取城市名称（第一个 term）和州名称（第二个 term，若存在）
                        val city = termsArray.getJSONObject(0).getString("value")
                        val state = if (termsArray.length() > 1) {
                            termsArray.getJSONObject(1).getString("value")
                        } else {
                            ""
                        }

                        // 格式化为 "City, State" 或仅 "City"
                        val formattedSuggestion = if (state.isNotEmpty()) {
                            "$city, $state"
                        } else {
                            city
                        }

                        suggestions.add(formattedSuggestion) // 添加到结果列表
                    }
                    continuation.resume(suggestions) // 恢复协程并返回结果
                } catch (e: Exception) {
                    e.printStackTrace()
                    continuation.resumeWithException(e) // 发生异常时恢复协程并抛出异常
                }
            },
            { error ->
                error.printStackTrace()
                continuation.resumeWithException(error) // 网络请求失败时恢复协程
            }
        )

        requestQueue.add(request)

        continuation.invokeOnCancellation {
            requestQueue.cancelAll { it == request } // 取消与该请求相关的所有操作
        }
    }
}


@Composable
fun CurrentWeatherCard(
    temperature: Float,
    summary: String,
    city: String,
    onClick: () -> Unit,
    image_id: Int,
) {
    println(image_id)
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(130.dp)
            .clickable { onClick() },
        shape = RectangleShape,
        colors = CardDefaults.cardColors(containerColor = Color(0xFF262626)),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically, // 垂直居中对齐
            horizontalArrangement = Arrangement.Center // 水平间距
        ) {
            // 天气图标
            Icon(
                painter = painterResource(id = image_id), // 替换为实际图标资源
                contentDescription = null,
                tint = Color.Unspecified,
                modifier = Modifier.size(60.dp)
            )

            Spacer(modifier = Modifier.width(16.dp))

            // 温度和天气描述的垂直布局
            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp) // 垂直间距
            ) {
                // 温度
                Text(
                    text = "${temperature.toInt()}°F",
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontSize = 23.sp,
                        fontWeight = FontWeight.Bold
                    ),
                    color = Color(0xFFbfbfbf)
                )

                // 天气描述
                Text(
                    text = summary,
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    ),
                    color = Color(0xFFbfbfbf)
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth() // 占满宽度
                .padding(top = 18.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically, // Row 内部子组件垂直居中
        ) {
            // 左侧的 Spacer，占据 1/3 的空间
            Spacer(modifier = Modifier.weight(1f))

            // 居中的城市名
            Text(
                text = city,
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontSize = 13.sp,
                ),
                color = Color(0xFFbfbfbf),
                modifier = Modifier.align(Alignment.CenterVertically)
            )

            // 右侧的 Spacer，占据 1/3 的空间
            Spacer(modifier = Modifier.weight(1f))

            // 图标
            Icon(
                painter = painterResource(id = com.example.myapplication.R.drawable.baseline_info_24),
                contentDescription = null,
                modifier = Modifier.align(Alignment.CenterVertically).padding(end = 8.dp) // 确保垂直方向居中
            )
        }
    }
}


@Composable
fun WeatherDetailsCard(
    humidity: Float,
    windSpeed: Float,
    visibility: Float,
    pressure: Float
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(110.dp),
        shape = RectangleShape,
        colors = CardDefaults.cardColors(containerColor = Color(0xFF262626))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp, start = 16.dp, end = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            WeatherDetailItem(
                icon = com.example.myapplication.R.drawable.humidity, // 替换为实际图标资源
                value = "${(humidity).toInt()}%",
                label = "Humidity"
            )
            WeatherDetailItem(
                icon = com.example.myapplication.R.drawable.wind_speed,
                value = "${"%.2f".format(windSpeed)} mph",
                label = "Wind Speed"
            )
            WeatherDetailItem(
                icon = com.example.myapplication.R.drawable.visibility,
                value = "${"%.2f".format(visibility)} mi",
                label = "Visibility"
            )
            WeatherDetailItem(
                icon = com.example.myapplication.R.drawable.pressure,
                value = "${"%.2f".format(pressure)} inHg",
                label = "Pressure"
            )
        }
    }
}

@Composable
fun WeatherDetailItem(icon: Int, value: String, label: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(
            painter = painterResource(id = icon),
            contentDescription = null,
            modifier = Modifier.size(45.dp),
            tint = Color.Black
        )

        Spacer(modifier = Modifier.height(5.dp))

        Text(text = value,
            style = MaterialTheme.typography.bodyMedium.copy(
                fontSize = 17.sp,
                fontWeight = FontWeight.Bold
            ),
            color = Color(0xFFbfbfbf))

        Text(text = label,
            style = MaterialTheme.typography.bodyMedium.copy(
                fontSize = 15.sp,
            ),
            color = Color(0xFFbfbfbf))
    }


}

@Composable
fun WeeklyForecastCard(dailyWeather: List<DailyWeather>) {
    println(dailyWeather)
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, end = 8.dp),
        shape = RectangleShape,
        colors = CardDefaults.cardColors(containerColor = Color(0xFF262626))
    ) {
        Column {
            dailyWeather.take(dailyWeather.size).forEachIndexed { index, weather ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(start = 8.dp, end = 8.dp, top = 16.dp, bottom = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween // 保留 SpaceBetween 以分配 Row 的空间
                ) {
                    // 日期
                    Text(
                        text = formatDate(weather.time),
                        modifier = Modifier
                            .padding(start = 8.dp)
                            .weight(1f), // 左侧占用更大空间
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = Color(0xFFbfbfbf)
                    )

                    // 天气图标
                    Icon(
                        painter = painterResource(id = weather.weather_image),
                        contentDescription = null,
                        modifier = Modifier
                            .size(24.dp)
                            .weight(0.5f), // 图标占较少空间
                        tint = Color.Unspecified
                    )

                    // 最低温度
                    Text(
                        text = "${weather.temperatureLow}",
                        modifier = Modifier.weight(0.5f), // 最低温度列
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = Color(0xFFbfbfbf)
                    )

                    // 最高温度
                    Text(
                        text = "${weather.temperatureHigh}",
                        modifier = Modifier
                            .padding(end = 8.dp)
                            .weight(0.5f), // 最高温度列
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontSize = 17.sp,
                            fontWeight = FontWeight.Bold
                        ),
                        color = Color(0xFFbfbfbf)
                    )
                }

                // 添加分割线（最后一行不需要分割线）
                if (index < dailyWeather.size - 1) {
                    Divider(
                        color = Color(0xFFbfbfbf),
                        thickness = 1.dp
                    )
                }
            }
        }
    }
}

// 工具函数：将时间戳转换为 YYYY-MM-DD 格式
fun formatDate(timestamp: String): String {
    return try {
        // 将字符串时间戳转换为 Long 类型
        val timeInMillis = timestamp.toLong() * 1000 // 秒转为毫秒
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()) // 定义日期格式
        sdf.format(Date(timeInMillis)) // 格式化日期
    } catch (e: Exception) {
        e.printStackTrace()
        "Invalid Date" // 如果解析失败，返回默认日期
    }
}