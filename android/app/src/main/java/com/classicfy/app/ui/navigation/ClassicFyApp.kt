package com.classicfy.app.ui.navigation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.calculateEndPadding
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.dp
import androidx.navigation.NavType
import androidx.navigation.navArgument
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.classicfy.app.ui.screen.login.LoginScreen
import com.classicfy.app.ui.screen.performance.PerformanceSelectionScreen
import com.classicfy.app.ui.screen.preference.PreferenceScreen
import com.classicfy.app.ui.screen.preference.PreferenceResultScreen
import com.classicfy.app.ui.screen.search.WorkSearchScreen
import com.classicfy.app.ui.screen.signup.SignUpScreen
import com.classicfy.app.ui.screen.splash.SplashScreen
import kotlinx.coroutines.delay

@Composable
fun ClassicFyApp() {
    val navController = rememberNavController()
    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    val layoutDirection = LocalLayoutDirection.current

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = ClassicFyDestination.SPLASH,
            modifier = Modifier
                .fillMaxSize()
                .padding(
                    start = innerPadding.calculateStartPadding(layoutDirection),
                    top = innerPadding.calculateTopPadding(),
                    end = innerPadding.calculateEndPadding(layoutDirection),
                    bottom = if (currentBackStackEntry?.destination?.route in setOf(
                            ClassicFyDestination.WORK_SEARCH,
                            ClassicFyDestination.PERFORMANCE_SELECTION
                        )
                    ) {
                        0.dp
                    } else {
                        innerPadding.calculateBottomPadding()
                    }
                )
        ) {
            composable(ClassicFyDestination.SPLASH) {
                SplashScreen()
                LaunchedEffect(Unit) {
                    delay(1_000L)
                    navController.navigate(ClassicFyDestination.LOGIN) {
                        popUpTo(ClassicFyDestination.SPLASH) { inclusive = true }
                        launchSingleTop = true
                    }
                }
            }
            composable(ClassicFyDestination.LOGIN) {
                LoginScreen(
                    onLoginClick = {
                        navController.navigate(ClassicFyDestination.PREFERENCE) {
                            launchSingleTop = true
                        }
                    },
                    onSignUpClick = {
                        navController.navigate(ClassicFyDestination.SIGN_UP) {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable(ClassicFyDestination.SIGN_UP) {
                SignUpScreen(
                    onSignUpClick = {
                        navController.popBackStack(ClassicFyDestination.LOGIN, false)
                    }
                )
            }
            composable(ClassicFyDestination.PREFERENCE) {
                PreferenceScreen(
                    onAnalyzeClick = {
                        navController.navigate(ClassicFyDestination.PREFERENCE_RESULT) {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable(ClassicFyDestination.PREFERENCE_RESULT) {
                PreferenceResultScreen(
                    onSearchClick = {
                        navController.navigate(ClassicFyDestination.WORK_SEARCH) {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable(ClassicFyDestination.WORK_SEARCH) {
                WorkSearchScreen(
                    onWorkClick = { workId ->
                        navController.navigate(ClassicFyDestination.performanceSelection(workId)) {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable(
                route = ClassicFyDestination.PERFORMANCE_SELECTION,
                arguments = listOf(navArgument("workId") { type = NavType.StringType })
            ) { backStackEntry ->
                PerformanceSelectionScreen(
                    workId = requireNotNull(backStackEntry.arguments?.getString("workId")),
                    onBackClick = { navController.popBackStack() },
                    onPerformanceClick = {
                        navController.navigate(ClassicFyDestination.DETAIL) {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable(ClassicFyDestination.DETAIL) {
                NavigationPlaceholder(
                    onAction = { navController.popBackStack() }
                )
            }
        }
    }
}

// Temporary content for checking navigation until product screens are added.
@Composable
private fun NavigationPlaceholder(
    onAction: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Detail placeholder")
        Button(onClick = onAction) {
            Text(text = "Back")
        }
    }
}
