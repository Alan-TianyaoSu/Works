package com.example.myapplication

import android.content.Context
import android.content.Intent
import android.graphics.drawable.AnimationDrawable
import android.net.Uri
import android.view.LayoutInflater
import android.widget.ImageView
import android.widget.ProgressBar
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.example.myapplication.viewmodels.WeatherViewModel
import com.example.myapplication.models.WeatherData

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import com.highsoft.highcharts.common.HIColor
import com.highsoft.highcharts.common.HIGradient
import com.highsoft.highcharts.common.HIStop

import com.highsoft.highcharts.core.*;
import com.highsoft.highcharts.common.hichartsclasses.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList
import androidx.compose.runtime.LaunchedEffect
import androidx.core.content.ContextCompat

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailedPage(
    weatherViewModel: WeatherViewModel,
    onBackClick: () -> Unit
) {
    val weatherDatum by weatherViewModel.weather.collectAsState()
    val searchFlag by weatherViewModel.isSearch.collectAsState()
    val weatherList by weatherViewModel.weathers.collectAsState()
    val pageNumber by weatherViewModel.pageNumber.collectAsState()

    val weatherData = if (searchFlag) weatherDatum else weatherList[pageNumber]

    // 当前选中的 Tab 索引
    var selectedTabIndex by remember { mutableStateOf(0) }

    // 短暂的加载状态
    var isTabLoading by remember { mutableStateOf(false) }

    // 创建一个 `Handler`
    val handler = remember { android.os.Handler(android.os.Looper.getMainLooper()) }

    Scaffold(
        containerColor = Color(0xFF000000),
        topBar = {
            TopAppBar(
                title = { Text(text = weatherData?.city ?: "NULL") },
                navigationIcon = {
                    IconButton(onClick = { onBackClick() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    val context = LocalContext.current
                    IconButton(onClick = { shareOnTwitter(context, weatherData) }) {
                        Icon(
                            painter = painterResource(id = R.drawable.twitter),
                            contentDescription = "X",
                            modifier = Modifier.size(32.dp),
                            tint = Color.Unspecified
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF000000),
                    titleContentColor = Color.White,
                    navigationIconContentColor = Color.White
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(Color(0xFF000000)),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // TabRowSection
            TabRowSection(
                selectedTabIndex = selectedTabIndex,
                onTabSelected = { index ->
                    // 切换 Tab 时触发加载状态
                    isTabLoading = true
                    selectedTabIndex = index
                    // 自动在 1 秒后关闭加载状态
                    handler.postDelayed({
                        isTabLoading = false
                    }, 1000)
                }
            )

            // 内容显示
            Box(
                modifier = Modifier.fillMaxSize()
            ) {
                // 实际内容
                when (selectedTabIndex) {
                    0 -> WeatherDetailsGrid(weatherData) // Today 的内容
                    1 -> WeeklyTabContent(weatherData)   // Weekly 的内容
                    2 -> WeatherDataTabContent(weatherData) // Weather Data 的内容
                }

                // 加载界面（覆盖在内容之上）
                if (isTabLoading) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Color(0xFF111111)) // 半透明黑色背景
                            .zIndex(1f), // 确保覆盖内容
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally, // 居中对齐
                            verticalArrangement = Arrangement.Center
                        ) {
                            // 使用 AndroidView 加载原生动画
                            AndroidView(
                                factory = { context ->
                                    ProgressBar(context).apply {
                                        indeterminateDrawable = ContextCompat.getDrawable(
                                            context,
                                            R.drawable.spinner // 动画资源
                                        )
                                    }
                                },
                                modifier = Modifier.size(76.dp) // 设置大小
                            )

                            // 间距
                            Spacer(modifier = Modifier.height(16.dp))

                            // 下方文字
                            Text(
                                text = "Fetching weather",
                                color = Color.White,
                            )
                        }
                    }
                }
            }
        }
    }
}

// 非 @Composable 的 Share 函数
private fun shareOnTwitter(context: Context, weatherData: WeatherData?) {
    try {

        val message = "Check Out ${weatherData?.city}'s Weather! It is ${weatherData?.temperature}°F! #CSCI571WeatherSearch"
        val twitterUrl = "https://x.com/intent/tweet?text=${Uri.encode(message)}"
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(twitterUrl))
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

@Composable
fun TabRowSection(
    selectedTabIndex: Int,          // 当前选中的 Tab 索引
    onTabSelected: (Int) -> Unit   // Tab 切换时的回调函数
) {
    // Tab 的标题和图标
    val tabs = listOf(
        Pair("Today", R.drawable.today),
        Pair("Weekly", R.drawable.weekly_tab),
        Pair("Weather Data", R.drawable.ic_thermometer)
    )

    TabRow(
        selectedTabIndex = selectedTabIndex,
        containerColor = Color(0xFF111111), // TabRow 背景颜色为黑色
        contentColor = Color.White,        // Tab 的内容颜色
        indicator = { tabPositions ->
            TabRowDefaults.Indicator(
                modifier = Modifier
                    .tabIndicatorOffset(tabPositions[selectedTabIndex]), // 根据选中的 Tab 索引定位
                height = 2.dp,                                          // 指示器高度
                color = Color.White                                     // 指示器颜色
            )
        },
        divider = {} // 不显示默认的分隔线
    ) {
        tabs.forEachIndexed { index, tab ->
            Tab(
                selected = selectedTabIndex == index,
                onClick = { onTabSelected(index) }, // 点击后触发回调，更新状态
                selectedContentColor = Color.White,  // 选中内容颜色为白色
                unselectedContentColor = Color.Gray, // 未选中内容颜色为灰色
                modifier = Modifier.padding(top = 10.dp, bottom = 10.dp)
            ) {
                val isSelected = selectedTabIndex == index // 当前 Tab 是否被选中
                val contentColor = if (isSelected) Color.White else Color.Gray // 动态设置颜色

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally, // 水平居中
                    modifier = Modifier.padding(8.dp).height(60.dp)
                ) {
                    Icon(
                        painter = painterResource(id = tab.second),
                        contentDescription = null,
                        modifier = Modifier.size(32.dp),
                        tint = contentColor // 动态设置图标的颜色
                    )
                    Spacer(modifier = Modifier.height(10.dp)) // 图标和文字之间的间距
                    Text(
                        text = tab.first,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontSize = 14.sp,
                            color = contentColor // 动态设置文字颜色
                        )
                    )
                }
            }
        }
    }
}

@Composable
fun WeatherDetailsGrid(weatherData: WeatherData?) {
    // 使用 LazyVerticalGrid 构建网格布局
    LazyVerticalGrid(
        columns = GridCells.Fixed(3), // 每行 3 列
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF2d2d2d)).padding(1.dp),
        horizontalArrangement = Arrangement.spacedBy(1.dp),
        verticalArrangement = Arrangement.spacedBy(1.dp),
    ) {
        // 显示天气详情
        item { WeatherDetailCard("Wind Speed", "${weatherData?.windSpeed} mph", R.drawable.wind_speed)}
        item { WeatherDetailCard("Pressure", "${weatherData?.pressure} inHg", R.drawable.pressure) }
        item { WeatherDetailCard("Precipitation", "${weatherData?.precipitation} %", R.drawable.rain_card) }
        item { WeatherDetailCard("Temperature", "${weatherData?.temperature}°F", R.drawable.ic_thermometer) }
        item { WeatherDetailCard(" ", weatherData?.weather_name ?: "NULL", weatherData?.weather_image ?: R.drawable.ic_launcher_background) }
        item { WeatherDetailCard("Humidity", "${weatherData?.humidity}%", R.drawable.humidity) }
        item { WeatherDetailCard("Visibility", "${weatherData?.visibility} mi", R.drawable.visibility) }
        item { WeatherDetailCard("Cloud Cover", "${weatherData?.cloud_cover} %", R.drawable.cloud_cover) }
        item { WeatherDetailCard("Ozone", "${weatherData?.ozone} ", R.drawable.uv) }
    }
}

@Composable
fun WeatherDetailCard(title: String, value: String, drawableResId: Int) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(4.dp).height(160.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1d1d1d)),
        shape = RoundedCornerShape(
            topStart = 8.dp, // 上左圆角
            topEnd = 8.dp,   // 上右圆角
            bottomStart = 0.dp, // 下左直角
            bottomEnd = 0.dp    // 下右直角
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 24.dp), // 外部边距
            verticalArrangement = Arrangement.spacedBy(16.dp), // 顶部 Icon 和底部 Text 保持上下分布
            horizontalAlignment = Alignment.CenterHorizontally // 水平居中
        ) {
            // 图标在顶部
            Icon(
                painter = painterResource(id = drawableResId),
                contentDescription = title,
                modifier = Modifier.size(64.dp),
                tint = Color.Unspecified
            )

            // 两行文本在底部
            Column(
                horizontalAlignment = Alignment.CenterHorizontally // 两行文本水平居中
            ) {
                Text(
                    text = value,
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontSize = 14.sp,
//                        fontWeight = FontWeight.Bold
                    ),
                    color = Color(0xFFbfbfbf)
                )
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyMedium.copy(
                        fontSize = 14.sp,
//                        fontWeight = FontWeight.Bold
                    ),
                    color = Color(0xFFbfbfbf)
                )
            }
        }
    }
}


@Composable
fun WeeklyTabContent(weatherData: WeatherData?) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF111111)), // 背景颜色
        horizontalAlignment = Alignment.CenterHorizontally // 水平居中
    ) {
        // 标题
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f), // 添加边框
            contentAlignment = Alignment.Center // 控制子项居中
        ) {
            Text(
                text = "Temperature Range",
                color = Color(0xFFbfbfbf),
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontSize = 24.sp,
//                    fontWeight = FontWeight.Bold
                ),
                textAlign = TextAlign.Center
            )
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(550.dp) // 设置图表高度
        ) {
            HighchartsView(weatherData)
        }
    }
}

@Composable
fun HighchartsView(weatherData: WeatherData?, modifier: Modifier = Modifier) {
    AndroidView(
        factory = { context ->
            // 加载 XML 布局
            val view = LayoutInflater.from(context).inflate(R.layout.highchart_view, null, false)
            val chartView = view.findViewById<HIChartView>(R.id.hc)

            // 配置 Highcharts 图表
            val seriesData = convertWeatherDataToSeries(weatherData)
            chartView.options = createHighchartsOptions(seriesData)

            view
        },
        modifier = modifier
            .fillMaxSize()
    )
}

fun createHighchartsOptions(seriesData: Array<Array<Any>>): HIOptions {
    val options = HIOptions()

    // 配置图表
    val chart = HIChart()
    chart.type = "arearange"
    val zooming = HIZooming()
    chart.zooming = zooming
    options.chart = chart

    // 标题
    val title = HITitle()
    title.text = "Temperature variation by day"
    options.title = title

    // X 轴
    val xAxis = HIXAxis()
    xAxis.type = "datetime"
    xAxis.crosshair = HICrosshair() // 启用十字准线
    val accessibility = HIAccessibility()
    xAxis.accessibility = accessibility
    options.xAxis = ArrayList<HIXAxis>().apply { add(xAxis) }

    // Y 轴
    val yAxis = HIYAxis()

    val yAxisTitle = HITitle()
    yAxisTitle.text = "Values" // 或者设置成你需要的标题文本
    yAxis.title = yAxisTitle

    options.yAxis = ArrayList<HIYAxis>().apply { add(yAxis) }

    // 提示框
    val tooltip = HITooltip()
    tooltip.shared = true
    tooltip.valueSuffix = "°F"
    tooltip.xDateFormat = "%A, %b %e"
    options.tooltip = tooltip

    // 图例
    val legend = HILegend()
    legend.enabled = false
    options.legend = legend

    // 数据序列
    val series = HIArearange()
    series.name = "Temperatures"
    series.data = ArrayList(Arrays.asList(*seriesData))

    // 创建渐变方向
    val gradient = HIGradient(0.0F, 0.0F, 0.0F, 1.0F) // 从上到下的渐变方向

    // 创建渐变的颜色停点
    val stops = LinkedList<HIStop>()

    // 起始颜色停点
    stops.add(HIStop(0.0F, HIColor.initWithRGBA(247, 163, 92, 1.0))) // 等价于 #f7a35c

    // 结束颜色停点
    stops.add(HIStop(1.0F, HIColor.initWithRGBA(124, 181, 236, 1.0))) // 等价于 #7cb5ec

    // 创建渐变颜色
    val color = HIColor.initWithLinearGradient(gradient, stops)
    series.color = color // 应用渐变颜色到数据序列
    series.lineColor = HIColor.initWithHexValue("#f7a35c")

    // 配置标记点样式
    val marker = HIMarker()
    marker.enabled = true
    marker.fillColor = HIColor.initWithRGBA(124, 181, 236, 1.0)
    marker.lineWidth = 2
    marker.lineColor = HIColor.initWithRGBA(124, 181, 236, 1.0)
    marker.radius = 2
    series.marker = marker

    options.series = ArrayList<HISeries>().apply { add(series) }

    val exporting = HIExporting()
    exporting.enabled = false // 完全禁用导出按钮
    options.exporting = exporting

    return options
}

// 将 WeatherData 转换为 Highcharts 的数据格式
fun convertWeatherDataToSeries(weatherData: WeatherData?): Array<Array<Any>> {
    return weatherData?.daily?.map { day ->
        // 将 Unix 时间戳 (秒) 转换为毫秒
        val date = day.time.toLongOrNull()?.times(1000) ?: 0L

        val tempLow = day.temperatureLow.toFloatOrNull() ?: 0.0f
        val tempHigh = day.temperatureHigh.toFloatOrNull() ?: 0.0f
        arrayOf<Any>(date, tempLow, tempHigh)
    }?.toTypedArray() ?: emptyArray()
}


@Composable
fun WeatherDataTabContent(weatherData: WeatherData?) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF111111)), // 背景颜色
        horizontalAlignment = Alignment.CenterHorizontally // 水平居中
    ) {
        // 标题
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f), // 添加边框
            contentAlignment = Alignment.Center // 控制子项居中
        ) {
            Text(
                text = "Weather Data",
                color = Color(0xFFbfbfbf),
                style = MaterialTheme.typography.bodyMedium.copy(
                    fontSize = 24.sp,
//                    fontWeight = FontWeight.Bold
                ),
                textAlign = TextAlign.Center
            )
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(550.dp) // 设置图表高度
        ) {
            HighchartsWeatherData(weatherData)
        }
    }
}

@Composable
fun HighchartsWeatherData(weatherData: WeatherData?, modifier: Modifier = Modifier) {
    AndroidView(
        factory = { context ->
            // 加载 XML 布局
            val view = LayoutInflater.from(context).inflate(R.layout.highchart_view, null, false)
            val chartView = view.findViewById<HIChartView>(R.id.hc)

            // 配置 Highcharts 图表
            val seriesData = convertWeatherData(weatherData)
            chartView.options = createWeatherDataOptions(seriesData)

            view
        },
        modifier = modifier
            .fillMaxSize()
    )
}

fun createWeatherDataOptions(seriesData: Map<String, Float>): HIOptions {
    val options = HIOptions()

    val chart = HIChart()
    chart.type = "solidgauge"
    chart.events = HIEvents().apply {
        render = HIFunction(
            "function renderIcons() { if (!this.series[0].icon) { this.series[0].icon = this.renderer.path(['M', -8, 0, 'L', 8, 0, 'M', 0, -8, 'L', 8, 0, 0, 8]).attr({'stroke': '#303030', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': 2, 'zIndex': 10}).add(this.series[2].group); } this.series[0].icon.translate(this.chartWidth / 2 - 10, this.plotHeight / 2 - this.series[0].points[0].shapeArgs.innerR - (this.series[0].points[0].shapeArgs.r - this.series[0].points[0].shapeArgs.innerR) / 2); if (!this.series[1].icon) { this.series[1].icon = this.renderer.path(['M', -8, 0, 'L', 8, 0, 'M', 0, -8, 'L', 8, 0, 0, 8, 'M', 8, -8, 'L', 16, 0, 8, 8]).attr({'stroke': '#ffffff', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': 2, 'zIndex': 10}).add(this.series[2].group); } this.series[1].icon.translate(this.chartWidth / 2 - 10, this.plotHeight / 2 - this.series[1].points[0].shapeArgs.innerR - (this.series[1].points[0].shapeArgs.r - this.series[1].points[0].shapeArgs.innerR) / 2); if (!this.series[2].icon) { this.series[2].icon = this.renderer.path(['M', 0, 8, 'L', 0, -8, 'M', -8, 0, 'L', 0, -8, 8, 0]).attr({'stroke': '#303030', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-width': 2, 'zIndex': 10}).add(this.series[2].group); } this.series[2].icon.translate(this.chartWidth / 2 - 10, this.plotHeight / 2 - this.series[2].points[0].shapeArgs.innerR - (this.series[2].points[0].shapeArgs.r - this.series[2].points[0].shapeArgs.innerR) / 2); }"
        )
    }
    options.chart = chart

    val title = HITitle()
    title.text = "Stat Summary"
    options.title = title

    // 配置 Pane（背景环形区域）
    val pane = HIPane().apply {
        startAngle = 0
        endAngle = 360
        background = ArrayList<HIBackground>().apply {
            add(HIBackground().apply {
                outerRadius = "112%"
                innerRadius = "88%"
                backgroundColor = HIColor.initWithRGBA(144, 238, 144, 0.35) // 浅绿色
                borderWidth = 0
            })
            add(HIBackground().apply {
                outerRadius = "87%"
                innerRadius = "63%"
                backgroundColor = HIColor.initWithRGBA(135, 206, 235, 0.35) // 浅蓝色
                borderWidth = 0
            })
            add(HIBackground().apply {
                outerRadius = "62%"
                innerRadius = "38%"
                backgroundColor = HIColor.initWithRGBA(255, 182, 193, 0.35) // 淡红色
                borderWidth = 0
            })
        }
    }
    options.pane = ArrayList<HIPane>().apply { add(pane) }

    // 配置 Y 轴
    val yAxis = HIYAxis().apply {
        min = 0
        max = 100
        lineWidth = 0
        tickPositions = ArrayList() // 移除刻度
    }
    options.yAxis = ArrayList<HIYAxis>().apply { add(yAxis) }

    // 配置工具提示（tooltip）
    val tooltip = HITooltip().apply {
        borderWidth = 0
        backgroundColor = HIColor.initWithName("none")
        style = HICSSObject().apply { fontSize = "16px" }
        pointFormat =
            "{series.name}<br><span style=\"font-size:2em; color: {point.color}; font-weight: bold\">{point.y}%</span>"
        positioner = HIFunction(
            "function (labelWidth) {" +
                    "   return {" +
                    "       x: (this.chart.chartWidth - labelWidth) / 2," +
                    "       y: (this.chart.plotHeight / 2) + 15" +
                    "   };" +
                    "}"
        )
    }
    options.tooltip = tooltip

    // 配置 PlotOptions
    val plotOptions = HIPlotOptions().apply {
        solidgauge = HISolidgauge().apply {
            dataLabels = ArrayList<HIDataLabels>().apply {
                add(HIDataLabels().apply { enabled = false })
            }
            linecap = "round"
            stickyTracking = false
            rounded = true
        }
    }
    options.plotOptions = plotOptions

    // 配置 Series 数据
    val series = ArrayList<HISeries>().apply {
        seriesData.forEach { (key, value) ->
            add(HISolidgauge().apply {
                name = key
                data = ArrayList<HIData>().apply {
                    add(HIData().apply {
                        color = when (key) {
                            "Cloud Cover" -> HIColor.initWithRGB(144, 238, 144) // 浅绿色
                            "Precipitation" -> HIColor.initWithRGB(135, 206, 235) // 浅蓝色
                            "Humidity" -> HIColor.initWithRGB(254, 128, 92) // 淡红色
                            else -> HIColor.initWithRGB(200, 200, 200)
                        }
                        radius = when (key) {
                            "Cloud Cover" -> "112%"
                            "Precipitation" -> "87%"
                            "Humidity" -> "62%"
                            else -> "50%"
                        }
                        innerRadius = when (key) {
                            "Cloud Cover" -> "88%"
                            "Precipitation" -> "63%"
                            "Humidity" -> "38%"
                            else -> "25%"
                        }
                        y = value.toDouble()
                    })
                }
            })
        }
    }
    options.series = series

    return options
}

fun convertWeatherData(weatherData: WeatherData?): Map<String, Float> {
    return mapOf(
        "Cloud Cover" to (weatherData?.cloud_cover ?: 0f),
        "Precipitation" to (weatherData?.precipitation ?: 0f),
        "Humidity" to (weatherData?.humidity ?: 0f)
    )
}

