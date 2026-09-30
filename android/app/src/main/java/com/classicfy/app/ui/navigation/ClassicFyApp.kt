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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavType
import androidx.navigation.navArgument
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.classicfy.app.R
import com.classicfy.app.ui.screen.login.LoginScreen
import com.classicfy.app.ui.screen.login.MOCK_LOGIN_PASSWORD
import com.classicfy.app.ui.screen.my.MyPageScreen
import com.classicfy.app.ui.screen.my.PasswordChangeScreen
import com.classicfy.app.ui.screen.performance.PerformanceSelectionScreen
import com.classicfy.app.ui.screen.performance.PerformanceDetailScreen
import com.classicfy.app.ui.screen.performance.initialTastePerformanceIds
import com.classicfy.app.ui.screen.performance.mockTastePerformances
import com.classicfy.app.ui.screen.preference.PreferenceScreen
import com.classicfy.app.ui.screen.preference.PreferenceResultScreen
import com.classicfy.app.ui.screen.search.WorkSearchScreen
import com.classicfy.app.ui.screen.signup.SignUpScreen
import com.classicfy.app.ui.screen.signup.SignUpCompleteScreen
import com.classicfy.app.ui.screen.splash.SplashScreen
import com.classicfy.app.ui.screen.taste.TastePerformanceListScreen
import com.classicfy.app.ui.screen.taste.TasteScreen
import kotlinx.coroutines.delay

@Composable
fun ClassicFyApp() {
    val navController = rememberNavController()
    val layoutDirection = LocalLayoutDirection.current
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val openEmptyWorkSearch: () -> Unit = {
        focusManager.clearFocus(force = true)
        keyboardController?.hide()
        navController.navigate(ClassicFyDestination.WORK_SEARCH) {
            popUpTo(ClassicFyDestination.WORK_SEARCH) { inclusive = true }
            launchSingleTop = true
        }
    }
    val openTaste: () -> Unit = {
        focusManager.clearFocus(force = true)
        keyboardController?.hide()
        navController.navigate(ClassicFyDestination.TASTE) {
            popUpTo(ClassicFyDestination.WORK_SEARCH) { inclusive = false }
            launchSingleTop = true
        }
    }
    val openMyPage: () -> Unit = {
        focusManager.clearFocus(force = true)
        keyboardController?.hide()
        navController.navigate(ClassicFyDestination.MY_PAGE) {
            popUpTo(ClassicFyDestination.WORK_SEARCH) { inclusive = false }
            launchSingleTop = true
        }
    }
    val initialNickname = stringResource(R.string.my_mock_nickname)
    val initialId = stringResource(R.string.my_mock_id)
    var savedNickname by rememberSaveable { mutableStateOf(initialNickname) }
    var savedId by rememberSaveable { mutableStateOf(initialId) }
    var currentMockPassword by remember { mutableStateOf(MOCK_LOGIN_PASSWORD) }
    var favoritePerformanceIds by rememberSaveable {
        mutableStateOf(initialTastePerformanceIds)
    }
    val tastePerformances = mockTastePerformances(favoritePerformanceIds)
    val toggleFavorite: (String) -> Unit = { id ->
        favoritePerformanceIds = if (id in favoritePerformanceIds) {
            favoritePerformanceIds - id
        } else {
            favoritePerformanceIds + id
        }
    }

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        val bottomContentPadding = innerPadding.calculateBottomPadding()
        NavHost(
            navController = navController,
            startDestination = ClassicFyDestination.SPLASH,
            modifier = Modifier
                .fillMaxSize()
                .padding(
                    start = innerPadding.calculateStartPadding(layoutDirection),
                    top = innerPadding.calculateTopPadding(),
                    end = innerPadding.calculateEndPadding(layoutDirection),
                    bottom = 0.dp
                )
        ) {
            composable(ClassicFyDestination.SPLASH) {
                SplashScreen(modifier = Modifier.padding(bottom = bottomContentPadding))
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
                    },
                    validPassword = currentMockPassword,
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(ClassicFyDestination.SIGN_UP) {
                SignUpScreen(
                    onSignUpClick = { nickname ->
                        navController.navigate(ClassicFyDestination.signUpComplete(nickname)) {
                            launchSingleTop = true
                        }
                    },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(
                route = ClassicFyDestination.SIGN_UP_COMPLETE,
                arguments = listOf(navArgument("nickname") { type = NavType.StringType })
            ) { backStackEntry ->
                SignUpCompleteScreen(
                    nickname = requireNotNull(backStackEntry.arguments?.getString("nickname")),
                    onStartClick = {
                        navController.popBackStack(ClassicFyDestination.LOGIN, false)
                    },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(ClassicFyDestination.PREFERENCE) {
                PreferenceScreen(
                    onAnalyzeClick = {
                        navController.navigate(ClassicFyDestination.PREFERENCE_RESULT) {
                            launchSingleTop = true
                        }
                    },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(ClassicFyDestination.PREFERENCE_RESULT) {
                PreferenceResultScreen(
                    onSearchClick = {
                        navController.navigate(ClassicFyDestination.WORK_SEARCH) {
                            launchSingleTop = true
                        }
                    },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(ClassicFyDestination.WORK_SEARCH) {
                WorkSearchScreen(
                    onSearchTabClick = openEmptyWorkSearch,
                    onTasteTabClick = openTaste,
                    onMyTabClick = openMyPage,
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
                    onPerformanceClick = { performanceId ->
                        navController.navigate(ClassicFyDestination.performanceDetail(performanceId)) {
                            launchSingleTop = true
                        }
                    },
                    favoriteIds = favoritePerformanceIds,
                    onFavoriteToggle = toggleFavorite,
                    onSearchTabClick = openEmptyWorkSearch,
                    onTasteTabClick = openTaste,
                    onMyTabClick = openMyPage
                )
            }
            composable(ClassicFyDestination.TASTE) {
                TasteScreen(
                    performances = tastePerformances,
                    onViewAllClick = {
                        navController.navigate(ClassicFyDestination.TASTE_PERFORMANCE_LIST) {
                            launchSingleTop = true
                        }
                    },
                    onPerformanceClick = { performanceId ->
                        navController.navigate(ClassicFyDestination.performanceDetail(performanceId)) {
                            launchSingleTop = true
                        }
                    },
                    onSearchTabClick = openEmptyWorkSearch,
                    onMyTabClick = openMyPage
                )
            }
            composable(ClassicFyDestination.TASTE_PERFORMANCE_LIST) {
                TastePerformanceListScreen(
                    performances = tastePerformances,
                    onBackClick = { navController.popBackStack() },
                    onPerformanceClick = { performanceId ->
                        navController.navigate(ClassicFyDestination.performanceDetail(performanceId)) {
                            launchSingleTop = true
                        }
                    },
                    onRemoveFavorite = { id ->
                        favoritePerformanceIds = favoritePerformanceIds - id
                    },
                    onRestoreFavorite = { id ->
                        favoritePerformanceIds = favoritePerformanceIds + id
                    },
                    onSearchTabClick = openEmptyWorkSearch,
                    onTasteTabClick = {
                        navController.popBackStack(ClassicFyDestination.TASTE, false)
                    },
                    onMyTabClick = openMyPage
                )
            }
            composable(ClassicFyDestination.MY_PAGE) {
                MyPageScreen(
                    savedNickname = savedNickname,
                    savedId = savedId,
                    onSave = { nickname, id ->
                        savedNickname = nickname
                        savedId = id
                    },
                    onLogoutClick = {
                        focusManager.clearFocus(force = true)
                        keyboardController?.hide()
                        savedNickname = initialNickname
                        savedId = initialId
                        favoritePerformanceIds = initialTastePerformanceIds
                        navController.navigate(ClassicFyDestination.LOGIN) {
                            popUpTo(ClassicFyDestination.LOGIN) { inclusive = true }
                            launchSingleTop = true
                        }
                    },
                    onPasswordChangeClick = {
                        navController.navigate(ClassicFyDestination.PASSWORD_CHANGE) {
                            launchSingleTop = true
                        }
                    },
                    onSearchTabClick = openEmptyWorkSearch,
                    onTasteTabClick = openTaste
                )
            }
            composable(ClassicFyDestination.PASSWORD_CHANGE) {
                PasswordChangeScreen(
                    onBackClick = { navController.popBackStack() },
                    onSaveClick = { password ->
                        currentMockPassword = password
                        navController.popBackStack()
                    },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
            composable(
                route = ClassicFyDestination.PERFORMANCE_DETAIL,
                arguments = listOf(navArgument("performanceId") { type = NavType.StringType })
            ) { backStackEntry ->
                val performanceId = requireNotNull(backStackEntry.arguments?.getString("performanceId"))
                val openedFromTaste = navController.previousBackStackEntry?.destination?.route in setOf(
                    ClassicFyDestination.TASTE,
                    ClassicFyDestination.TASTE_PERFORMANCE_LIST
                )
                PerformanceDetailScreen(
                    performanceId = performanceId,
                    isFavorite = performanceId in favoritePerformanceIds,
                    onFavoriteToggle = { toggleFavorite(performanceId) },
                    onBackClick = { navController.popBackStack() },
                    onSearchClick = openEmptyWorkSearch,
                    activeBottomDestination = if (openedFromTaste) {
                        ClassicFyBottomDestination.TASTE
                    } else {
                        ClassicFyBottomDestination.SEARCH
                    },
                    onTasteClick = if (openedFromTaste) {
                        {
                            navController.popBackStack(ClassicFyDestination.TASTE, false)
                            Unit
                        }
                    } else {
                        openTaste
                    },
                    onMyClick = openMyPage
                )
            }
            composable(ClassicFyDestination.DETAIL) {
                NavigationPlaceholder(
                    onAction = { navController.popBackStack() },
                    modifier = Modifier.padding(bottom = bottomContentPadding)
                )
            }
        }
    }
}

// Temporary content for checking navigation until product screens are added.
@Composable
private fun NavigationPlaceholder(
    onAction: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
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
