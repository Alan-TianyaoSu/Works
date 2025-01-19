package com.example.myapplication

import android.content.Context
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Paint
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.myapplication.ui.theme.MyApplicationTheme
import com.example.myapplication.viewmodels.WeatherViewModel

import kotlinx.coroutines.launch
import android.graphics.Canvas
import android.graphics.Color
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import android.graphics.Paint
import androidx.appcompat.app.AppCompatDelegate

val googleApiKey = "AIzaSyBOI4I20qLdWyWRU3ha2r_SErsdCqTkCRg"
//val appKey = "TS1FKJFkwAECNDQ4V8D9ZCpIevYPwC4G"
val appKey = "rRy4klckCQk2bL8ijgtBbfvrhuJpuDzV"
//val appKey = "iWKlQCy58YuZp47ZT3g2BRC9GDfuvrKe"
//val appKey = "ja0wA1hYa6JNQI6RxNo27x7egPp8rUTT"



class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
//        setTheme(R.style.AppTheme)

        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);

        setContentView(R.layout.splash_screen)

        super.onCreate(savedInstanceState)

        // 2. 获取 WeatherViewModel
        val weatherViewModel = ViewModelProvider(this)[WeatherViewModel::class.java]

        // 3. 监听数据加载状态
        lifecycleScope.launchWhenStarted {
            weatherViewModel.isLoading.collect { isLoading ->
                if (!isLoading) {
                    // 数据加载完成后，切换到主界面
                    setContent {
                        MyApplicationTheme {
                            AppNavigation(weatherViewModel)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun AppNavigation(weatherViewModel: WeatherViewModel) {
    // 创建 NavController
    val navController = rememberNavController()

    // 使用 Scaffold 包装页面导航
    Scaffold(
        modifier = Modifier.fillMaxSize() // 确保占满屏幕
    ) { innerPadding ->
        // 定义导航主机
        NavHost(
            navController = navController,
            startDestination = "home" // 设置初始页面为 HomeScreen
        ) {
            // HomeScreen 路由
            composable("home") {
                HomeScreen(
                    weatherViewModel = weatherViewModel, // 传递 WeatherViewModel
                    onCardClick = {
                        // 点击事件：跳转到 DetailedPage
                        navController.navigate("detailed")
                    },
                    modifier = Modifier.padding(innerPadding) // 为页面内容添加内边距
                )
            }
            // DetailedPage 路由
            composable("detailed") {
                DetailedPage(
                    weatherViewModel = weatherViewModel, // 传递 WeatherViewModel
                    onBackClick = {
                        // 点击事件：返回 HomeScreen
                        navController.popBackStack()
                    }
                )
            }
        }
    }
}