package com.classicfy.app.ui.theme

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor

val ClassicFyBackground = Color(0xFF0D0D0D)
val ClassicFySurface = Color(0xFF1A1A1A)
val ClassicFySurfaceVariant = Color(0xFF252525)
val ClassicFyAccent = Color(0xFFC8A65A)
val ClassicFyGoldStart = Color(0xFFC9A85B)
val ClassicFyGoldMiddle = Color(0xFFD9C18B)
val ClassicFyGoldEnd = Color(0xFFCCB57E)

val ClassicFyGoldGradient = Brush.horizontalGradient(
    0f to ClassicFyGoldStart,
    0.73f to ClassicFyGoldMiddle,
    0.96f to ClassicFyGoldEnd
)

// Toggle this experiment off to restore the original solid yellow throughout the app.
private const val UseExperimentalGold = true
val ClassicFyActiveAccent = if (UseExperimentalGold) ClassicFyGoldEnd else ClassicFyAccent
val ClassicFyPrimaryButtonBrush: Brush =
    if (UseExperimentalGold) ClassicFyGoldGradient else SolidColor(ClassicFyAccent)

val ClassicFyOnAccent = Color(0xFF0D0D0D)
val ClassicFyTextPrimary = Color(0xFFF5F5F0)
val ClassicFyTextSecondary = Color(0xFFB8B8B3)
