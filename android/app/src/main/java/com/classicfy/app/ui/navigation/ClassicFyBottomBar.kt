package com.classicfy.app.ui.navigation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.classicfy.app.R

enum class ClassicFyBottomDestination {
    SEARCH,
    TASTE,
    MY
}

@Composable
fun ClassicFyBottomBar(
    activeDestination: ClassicFyBottomDestination,
    onSearchClick: () -> Unit,
    modifier: Modifier = Modifier,
    onTasteClick: (() -> Unit)? = null,
    onMyClick: (() -> Unit)? = null
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
    ) {
        HorizontalDivider(
            thickness = 1.dp,
            color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
        )
        NavigationBar(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding(),
            containerColor = MaterialTheme.colorScheme.surface,
            tonalElevation = 0.dp,
            // Insets are applied above while the surface paints through them.
            windowInsets = WindowInsets(0, 0, 0, 0)
        ) {
            val itemColors = NavigationBarItemDefaults.colors(
                selectedIconColor = MaterialTheme.colorScheme.primary,
                selectedTextColor = MaterialTheme.colorScheme.primary,
                indicatorColor = MaterialTheme.colorScheme.surface,
                unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant,
                disabledIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                disabledTextColor = MaterialTheme.colorScheme.onSurfaceVariant
            )
            NavigationBarItem(
                selected = activeDestination == ClassicFyBottomDestination.SEARCH,
                onClick = onSearchClick,
                icon = {
                    Icon(painterResource(R.drawable.ic_search), contentDescription = null)
                },
                label = { Text(stringResource(R.string.bottom_search)) },
                colors = itemColors
            )
            NavigationBarItem(
                selected = activeDestination == ClassicFyBottomDestination.TASTE,
                onClick = { onTasteClick?.invoke() },
                enabled = onTasteClick != null,
                icon = {
                    Icon(painterResource(R.drawable.ic_taste), contentDescription = null)
                },
                label = { Text(stringResource(R.string.bottom_taste)) },
                colors = itemColors
            )
            NavigationBarItem(
                selected = activeDestination == ClassicFyBottomDestination.MY,
                onClick = { onMyClick?.invoke() },
                enabled = onMyClick != null,
                icon = {
                    Icon(painterResource(R.drawable.ic_person), contentDescription = null)
                },
                label = { Text(stringResource(R.string.bottom_my)) },
                colors = itemColors
            )
        }
    }
}
