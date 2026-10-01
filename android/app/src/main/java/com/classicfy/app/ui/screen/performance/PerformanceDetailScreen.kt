package com.classicfy.app.ui.screen.performance

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.classicfy.app.R
import com.classicfy.app.ui.component.FavoriteRemovalNotice
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination

@Composable
fun PerformanceDetailScreen(
    performanceId: String,
    isFavorite: Boolean,
    showUndoOnRemoval: Boolean,
    onFavoriteToggle: () -> Unit,
    onRestoreFavorite: () -> Unit,
    onBackClick: () -> Unit,
    onSearchClick: () -> Unit,
    activeBottomDestination: ClassicFyBottomDestination,
    onTasteClick: () -> Unit,
    onMyClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val performance = remember(performanceId) {
        requireNotNull(mockPerformanceById(performanceId))
    }
    var showRemovalNotice by rememberSaveable(performanceId, showUndoOnRemoval) {
        mutableStateOf(false)
    }
    val undoRemoval = {
        onRestoreFavorite()
        showRemovalNotice = false
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        IconButton(
            onClick = onBackClick,
            modifier = Modifier.padding(start = 12.dp, top = 8.dp)
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_arrow_back),
                contentDescription = stringResource(R.string.performance_detail_back),
                tint = MaterialTheme.colorScheme.onBackground
            )
        }
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(start = 24.dp, top = 12.dp, end = 24.dp, bottom = 24.dp)
        ) {
            item {
                Text(
                    text = stringResource(R.string.performance_detail_title),
                    style = MaterialTheme.typography.titleLarge.copy(fontSize = 24.sp),
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(32.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "${performance.composer}: ${performance.work}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            text = performance.performer,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    IconButton(onClick = {
                        onFavoriteToggle()
                        if (isFavorite && showUndoOnRemoval) {
                            showRemovalNotice = true
                        }
                    }) {
                        Icon(
                            painter = painterResource(
                                if (isFavorite) R.drawable.ic_favorite_filled
                                else R.drawable.ic_taste
                            ),
                            contentDescription = stringResource(
                                if (isFavorite) R.string.performance_remove_favorite
                                else R.string.performance_add_favorite
                            ),
                            tint = if (isFavorite) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                Spacer(Modifier.height(36.dp))
                Text(
                    text = stringResource(R.string.performance_reason_rank, performance.rank),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(16.dp))
                Text(
                    text = performance.detailedReason,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 200.dp)
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(16.dp)
                        )
                        .padding(20.dp)
                )
            }
        }
        ClassicFyBottomBar(
            activeDestination = activeBottomDestination,
            onSearchClick = onSearchClick,
            onTasteClick = onTasteClick,
            onMyClick = onMyClick
        )
    }

    if (showUndoOnRemoval && showRemovalNotice) {
        FavoriteRemovalNotice(
            onUndo = undoRemoval,
            onConfirm = { showRemovalNotice = false }
        )
    }
}
