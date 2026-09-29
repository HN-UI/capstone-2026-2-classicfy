package com.classicfy.app.ui.screen.performance

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.classicfy.app.R
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination
import com.classicfy.app.ui.screen.search.Work
import com.classicfy.app.ui.screen.search.mockWorks

@Composable
fun PerformanceSelectionScreen(
    workId: String,
    onBackClick: () -> Unit,
    onPerformanceClick: (String) -> Unit,
    favoriteIds: Set<String>,
    onFavoriteToggle: (String) -> Unit,
    onSearchTabClick: () -> Unit,
    onTasteTabClick: () -> Unit,
    onMyTabClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedWorkId by rememberSaveable(workId) { mutableStateOf(workId) }
    val selectedWork = remember(selectedWorkId) {
        requireNotNull(mockWorks.find { it.id == selectedWorkId })
    }
    val performances = remember(selectedWorkId) { mockPerformancesFor(selectedWork) }
    var expandedPerformanceId by rememberSaveable(workId) { mutableStateOf<String?>(null) }
    var quickQuery by rememberSaveable(workId) { mutableStateOf("") }
    val suggestions = remember(quickQuery) {
        val term = quickQuery.trim()
        if (term.isEmpty()) emptyList() else mockWorks.filter { work ->
            work.composer.contains(term, ignoreCase = true) ||
                work.title.contains(term, ignoreCase = true)
        }.take(3)
    }
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val selectWork: (Work) -> Unit = { work ->
        selectedWorkId = work.id
        expandedPerformanceId = null
        quickQuery = ""
        focusManager.clearFocus()
        keyboardController?.hide()
    }
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .imePadding()
    ) {
        IconButton(
            onClick = onBackClick,
            modifier = Modifier.padding(start = 12.dp, top = 8.dp)
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_arrow_back),
                contentDescription = stringResource(R.string.performance_back),
                tint = MaterialTheme.colorScheme.onBackground
            )
        }
        Spacer(Modifier.height(12.dp))
        QuickWorkSearchField(
            query = quickQuery,
            onQueryChange = { quickQuery = it },
            onSearch = { suggestions.firstOrNull()?.let(selectWork) }
        )
        if (quickQuery.isNotBlank()) {
            Spacer(Modifier.height(4.dp))
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
                    .background(
                        MaterialTheme.colorScheme.surfaceVariant,
                        RoundedCornerShape(12.dp)
                    )
            ) {
                if (suggestions.isNotEmpty()) {
                    suggestions.forEach { work ->
                        QuickWorkSuggestion(work = work, onClick = { selectWork(work) })
                    }
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .padding(horizontal = 16.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = stringResource(R.string.work_search_no_suggestions),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }
        Spacer(Modifier.height(28.dp))
        Column(modifier = Modifier.padding(horizontal = 24.dp)) {
            Text(
                text = stringResource(R.string.performance_selected_work),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = "${selectedWork.composer}: ${selectedWork.title}",
                style = MaterialTheme.typography.titleMedium.copy(fontSize = 18.sp),
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Spacer(Modifier.height(32.dp))
            Text(
                text = stringResource(R.string.performance_recommendations),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onBackground
            )
        }
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 12.dp)
        ) {
            items(performances, key = { it.id }) { performance ->
                PerformanceRow(
                    performance = performance,
                    isFavorite = performance.id in favoriteIds,
                    isExpanded = expandedPerformanceId == performance.id,
                    onFavoriteClick = { onFavoriteToggle(performance.id) },
                    onToggleExpand = {
                        expandedPerformanceId = if (expandedPerformanceId == performance.id) {
                            null
                        } else {
                            performance.id
                        }
                    },
                    onDetailClick = { onPerformanceClick(performance.id) }
                )
            }
        }
        ClassicFyBottomBar(
            activeDestination = ClassicFyBottomDestination.SEARCH,
            onSearchClick = onSearchTabClick,
            onTasteClick = onTasteTabClick,
            onMyClick = onMyTabClick
        )
    }
}

@Composable
private fun QuickWorkSearchField(
    query: String,
    onQueryChange: (String) -> Unit,
    onSearch: () -> Unit
) {
    BasicTextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .height(48.dp)
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(14.dp)),
        singleLine = true,
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = { onSearch() }),
        textStyle = MaterialTheme.typography.bodyMedium.copy(
            color = MaterialTheme.colorScheme.onSurface
        ),
        cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
        decorationBox = { innerTextField ->
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_search),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(Modifier.width(12.dp))
                Box(modifier = Modifier.weight(1f)) {
                    if (query.isEmpty()) {
                        Text(
                            text = stringResource(R.string.work_search_placeholder),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    innerTextField()
                }
            }
        }
    )
}

@Composable
private fun QuickWorkSuggestion(work: Work, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            painter = painterResource(R.drawable.ic_search),
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.size(18.dp)
        )
        Spacer(Modifier.width(12.dp))
        Text(
            text = "${work.composer}: ${work.title}",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = "↖",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun PerformanceRow(
    performance: PerformanceItem,
    isFavorite: Boolean,
    isExpanded: Boolean,
    onFavoriteClick: () -> Unit,
    onToggleExpand: () -> Unit,
    onDetailClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = if (isExpanded) 8.dp else 0.dp)
            .background(
                if (isExpanded) MaterialTheme.colorScheme.surfaceVariant
                else MaterialTheme.colorScheme.background,
                RoundedCornerShape(16.dp)
            )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 72.dp)
                .clickable(onClick = onToggleExpand)
                .padding(start = if (isExpanded) 16.dp else 0.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "${performance.composer}: ${performance.work}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = performance.performer,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            IconButton(onClick = onFavoriteClick) {
                Icon(
                    painter = painterResource(
                        if (isFavorite) R.drawable.ic_favorite_filled else R.drawable.ic_taste
                    ),
                    contentDescription = stringResource(
                        if (isFavorite) R.string.performance_remove_favorite
                        else R.string.performance_add_favorite
                    ),
                    tint = if (isFavorite) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            IconButton(onClick = onToggleExpand) {
                Icon(
                    painter = painterResource(R.drawable.ic_chevron_right),
                    contentDescription = stringResource(
                        if (isExpanded) R.string.performance_collapse
                        else R.string.performance_expand
                    ),
                    tint = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier
                        .size(20.dp)
                        .rotate(if (isExpanded) -90f else 0f)
                )
            }
        }
        if (isExpanded) {
            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 16.dp),
                thickness = 1.dp,
                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
            )
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 16.dp)) {
                Text(
                    text = stringResource(R.string.performance_reason_rank, performance.rank),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = performance.shortReason,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 16.dp),
                thickness = 1.dp,
                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
            )
            TextButton(
                onClick = onDetailClick,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = stringResource(R.string.performance_view_details),
                    color = MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
    if (!isExpanded) {
        HorizontalDivider(
            thickness = 1.dp,
            color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
        )
    }
}
