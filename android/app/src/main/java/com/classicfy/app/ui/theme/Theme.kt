package com.classicfy.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ClassicFyColorScheme = darkColorScheme(
    primary = ClassicFyActiveAccent,
    onPrimary = ClassicFyOnAccent,
    primaryContainer = ClassicFySurfaceVariant,
    onPrimaryContainer = ClassicFyTextPrimary,
    inversePrimary = ClassicFyActiveAccent,
    secondary = ClassicFyActiveAccent,
    onSecondary = ClassicFyOnAccent,
    secondaryContainer = ClassicFySurfaceVariant,
    onSecondaryContainer = ClassicFyTextPrimary,
    tertiary = ClassicFyActiveAccent,
    onTertiary = ClassicFyOnAccent,
    tertiaryContainer = ClassicFySurfaceVariant,
    onTertiaryContainer = ClassicFyTextPrimary,
    background = ClassicFyBackground,
    onBackground = ClassicFyTextPrimary,
    surface = ClassicFySurface,
    onSurface = ClassicFyTextPrimary,
    surfaceVariant = ClassicFySurfaceVariant,
    onSurfaceVariant = ClassicFyTextSecondary,
    surfaceTint = ClassicFyActiveAccent,
    outline = ClassicFyTextSecondary
)

@Composable
fun ClassicFyTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = ClassicFyColorScheme,
        typography = ClassicFyTypography,
        content = content
    )
}
