package com.classicfy.app.ui.navigation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.classicfy.app.ui.screen.login.LoginScreen
import com.classicfy.app.ui.screen.preference.PreferenceScreen
import com.classicfy.app.ui.screen.preference.PreferenceResultScreen
import com.classicfy.app.ui.screen.signup.SignUpScreen
import com.classicfy.app.ui.screen.splash.SplashScreen
import kotlinx.coroutines.delay

@Composable
fun ClassicFyApp() {
    val navController = rememberNavController()

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = ClassicFyDestination.SPLASH,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
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
