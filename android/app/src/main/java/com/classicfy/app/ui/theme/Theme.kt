package com.classicfy.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ClassicFyColorScheme = darkColorScheme(
    primary = ClassicFyAccent,
    onPrimary = ClassicFyOnAccent,
    primaryContainer = ClassicFySurfaceVariant,
    onPrimaryContainer = ClassicFyTextPrimary,
    inversePrimary = ClassicFyAccent,
    secondary = ClassicFyAccent,
    onSecondary = ClassicFyOnAccent,
    secondaryContainer = ClassicFySurfaceVariant,
    onSecondaryContainer = ClassicFyTextPrimary,
    tertiary = ClassicFyAccent,
    onTertiary = ClassicFyOnAccent,
    tertiaryContainer = ClassicFySurfaceVariant,
    onTertiaryContainer = ClassicFyTextPrimary,
    background = ClassicFyBackground,
    onBackground = ClassicFyTextPrimary,
    surface = ClassicFySurface,
    onSurface = ClassicFyTextPrimary,
    surfaceVariant = ClassicFySurfaceVariant,
    onSurfaceVariant = ClassicFyTextSecondary,
    surfaceTint = ClassicFyAccent,
    outline = ClassicFyTextSecondary
)

@Composable
fun ClassicFyTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = ClassicFyColorScheme,
        typography = Typography,
        content = content
    )
}
