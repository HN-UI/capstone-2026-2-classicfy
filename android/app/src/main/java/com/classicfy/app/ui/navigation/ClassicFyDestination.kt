package com.classicfy.app.ui.navigation

import android.net.Uri

internal object ClassicFyDestination {
    const val SPLASH = "splash"
    const val LOGIN = "login"
    const val SIGN_UP = "sign_up"
    const val SIGN_UP_COMPLETE = "sign_up_complete/{nickname}"
    const val PREFERENCE = "preference"
    const val PREFERENCE_RESULT = "preference_result"
    const val WORK_SEARCH = "work_search"
    const val TASTE = "taste"
    const val TASTE_PERFORMANCE_LIST = "taste_performance_list"
    const val MY_PAGE = "my_page"
    const val PASSWORD_CHANGE = "password_change"
    const val PERFORMANCE_SELECTION = "performance_selection/{workId}"
    const val PERFORMANCE_DETAIL = "performance_detail/{performanceId}"
    const val DETAIL = "detail"

    fun performanceSelection(workId: String) = "performance_selection/$workId"
    fun performanceDetail(performanceId: String) = "performance_detail/$performanceId"
    fun signUpComplete(nickname: String) = "sign_up_complete/${Uri.encode(nickname)}"
}
