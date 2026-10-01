package com.classicfy.app.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.classicfy.app.R

// Register bundled families from res/font here, then change only this declaration
// to switch the font used by every Material typography style.
val ChosunNmFontFamily = FontFamily(
    Font(R.font.chosun_nm, FontWeight.Normal)
)

val GowunDodumFontFamily = FontFamily(
    Font(R.font.gowundodum_regular, FontWeight.Normal)
)

val RidiBatangFontFamily = FontFamily(
    Font(R.font.ridibatang, FontWeight.Normal)
)

val MaruBuriFontFamily = FontFamily(
    Font(R.font.maruburi_light, FontWeight.Light),
    Font(R.font.maruburi_regular, FontWeight.Normal),
    Font(R.font.maruburi_semibold, FontWeight.SemiBold)
)

val NanumSquareNeoFontFamily = FontFamily(
    Font(R.font.nanumsquareneo_alt, FontWeight.Light),
    Font(R.font.nanumsquareneo_brg, FontWeight.Normal),
    Font(R.font.nanumsquareneo_cbd, FontWeight.Bold)
)

val PretendardFontFamily = FontFamily(
    Font(R.font.pretendard_regular, FontWeight.Normal),
    Font(R.font.pretendard_medium, FontWeight.Medium),
    Font(R.font.pretendard_semibold, FontWeight.SemiBold)
)

val SuitFontFamily = FontFamily(
    Font(R.font.suit_regular, FontWeight.Normal),
    Font(R.font.suit_medium, FontWeight.Medium),
    Font(R.font.suit_semibold, FontWeight.SemiBold)
)

val ActiveFontFamily: FontFamily = PretendardFontFamily

private val MaterialDefaults = Typography()

val ClassicFyTypography = Typography(
    displayLarge = MaterialDefaults.displayLarge.copy(fontFamily = ActiveFontFamily),
    displayMedium = MaterialDefaults.displayMedium.copy(fontFamily = ActiveFontFamily),
    displaySmall = MaterialDefaults.displaySmall.copy(fontFamily = ActiveFontFamily),
    // Screen title
    headlineLarge = MaterialDefaults.headlineLarge.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 28.sp,
        fontWeight = FontWeight.Bold,
        lineHeight = 34.sp,
        letterSpacing = 0.sp
    ),
    headlineMedium = MaterialDefaults.headlineMedium.copy(fontFamily = ActiveFontFamily),
    headlineSmall = MaterialDefaults.headlineSmall.copy(fontFamily = ActiveFontFamily),
    // Section title and primary list item
    titleLarge = MaterialDefaults.titleLarge.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 20.sp,
        fontWeight = FontWeight.Bold,
        lineHeight = 26.sp,
        letterSpacing = 0.sp
    ),
    titleMedium = MaterialDefaults.titleMedium.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 17.sp,
        fontWeight = FontWeight.SemiBold,
        lineHeight = 24.sp,
        letterSpacing = 0.sp
    ),
    titleSmall = MaterialDefaults.titleSmall.copy(fontFamily = ActiveFontFamily),
    bodyLarge = TextStyle(
        fontFamily = ActiveFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.5.sp
    ),
    bodyMedium = MaterialDefaults.bodyMedium.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 14.sp,
        fontWeight = FontWeight.Normal,
        lineHeight = 20.sp,
        letterSpacing = 0.sp
    ),
    bodySmall = MaterialDefaults.bodySmall.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 14.sp,
        fontWeight = FontWeight.Normal,
        lineHeight = 20.sp,
        letterSpacing = 0.sp
    ),
    // Major actions, navigation captions, and validation messages
    labelLarge = MaterialDefaults.labelLarge.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 16.sp,
        fontWeight = FontWeight.SemiBold,
        lineHeight = 22.sp,
        letterSpacing = 0.sp
    ),
    labelMedium = MaterialDefaults.labelMedium.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
        lineHeight = 16.sp,
        letterSpacing = 0.sp
    ),
    labelSmall = MaterialDefaults.labelSmall.copy(
        fontFamily = ActiveFontFamily,
        fontSize = 14.sp,
        fontWeight = FontWeight.Medium,
        lineHeight = 20.sp,
        letterSpacing = 0.sp
    )
)
