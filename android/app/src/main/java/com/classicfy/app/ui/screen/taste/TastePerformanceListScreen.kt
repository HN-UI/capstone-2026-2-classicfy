package com.classicfy.app.ui.screen.taste

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
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.component.FavoriteRemovalNotice
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination
import com.classicfy.app.ui.screen.performance.PerformanceItem
import java.text.Collator
import java.util.Locale

private enum class TasteSortCriterion { ADDED_DATE, COMPOSER, WORK_TITLE }
private enum class TasteSortDirection { ASCENDING, DESCENDING }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun TastePerformanceListScreen(
    performances: List<PerformanceItem>,
    addedAtById: Map<String, Long>,
    onBackClick: () -> Unit,
    onPerformanceClick: (String) -> Unit,
    onRemoveFavorite: (String) -> Unit,
    onRestoreFavorite: (String) -> Unit,
    onSearchTabClick: () -> Unit,
    onTasteTabClick: () -> Unit,
    onMyTabClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var query by rememberSaveable { mutableStateOf("") }
    var pendingRemovedId by rememberSaveable { mutableStateOf<String?>(null) }
    var sortCriterion by rememberSaveable { mutableStateOf(TasteSortCriterion.ADDED_DATE) }
    var sortDirection by rememberSaveable { mutableStateOf(TasteSortDirection.DESCENDING) }
    var showSortSheet by rememberSaveable { mutableStateOf(false) }
    val filteredPerformances = remember(query, performances) {
        val term = query.trim()
        if (term.isEmpty()) performances else performances.filter { performance ->
            performance.composer.contains(term, ignoreCase = true) ||
                performance.work.contains(term, ignoreCase = true) ||
                performance.performer.contains(term, ignoreCase = true)
        }
    }
    val sortedPerformances = remember(filteredPerformances, addedAtById, sortCriterion, sortDirection) {
        val collator = Collator.getInstance(Locale.KOREAN)
        val ascending: Comparator<PerformanceItem> = when (sortCriterion) {
            TasteSortCriterion.ADDED_DATE ->
                compareBy<PerformanceItem>({ addedAtById[it.id] ?: 0L }, { it.id })
            TasteSortCriterion.COMPOSER -> Comparator { first, second ->
                val composerOrder = collator.compare(first.composer, second.composer)
                val workOrder = if (composerOrder == 0) collator.compare(first.work, second.work)
                else composerOrder
                if (workOrder == 0) first.id.compareTo(second.id) else workOrder
            }
            TasteSortCriterion.WORK_TITLE -> Comparator { first, second ->
                val workOrder = collator.compare(first.work, second.work)
                val composerOrder = if (workOrder == 0) collator.compare(first.composer, second.composer)
                else workOrder
                if (composerOrder == 0) first.id.compareTo(second.id) else composerOrder
            }
        }
        filteredPerformances.sortedWith(
            if (sortDirection == TasteSortDirection.ASCENDING) ascending else ascending.reversed()
        )
    }
    val sortLabelRes = when (sortCriterion) {
        TasteSortCriterion.ADDED_DATE -> if (sortDirection == TasteSortDirection.DESCENDING) {
            R.string.taste_sort_recent
        } else R.string.taste_sort_oldest
        TasteSortCriterion.COMPOSER -> if (sortDirection == TasteSortDirection.ASCENDING) {
            R.string.taste_sort_composer_ascending
        } else R.string.taste_sort_composer_descending
        TasteSortCriterion.WORK_TITLE -> if (sortDirection == TasteSortDirection.ASCENDING) {
            R.string.taste_sort_work_ascending
        } else R.string.taste_sort_work_descending
    }
    val undoRemoval = {
        pendingRemovedId?.let(onRestoreFavorite)
        pendingRemovedId = null
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
                contentDescription = stringResource(R.string.taste_list_back),
                tint = MaterialTheme.colorScheme.onBackground
            )
        }
        Spacer(Modifier.height(12.dp))
        TasteSearchField(query = query, onQueryChange = { query = it })
        Spacer(Modifier.height(28.dp))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = stringResource(R.string.taste_performance_count, filteredPerformances.size),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onBackground
            )
            if (filteredPerformances.isNotEmpty()) {
                Row(
                    modifier = Modifier
                        .clickable { showSortSheet = true }
                        .padding(vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = stringResource(sortLabelRes),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Icon(
                        painter = painterResource(R.drawable.ic_chevron_right),
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(16.dp).rotate(90f)
                    )
                }
            }
        }
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(start = 24.dp, top = 12.dp, end = 24.dp, bottom = 24.dp)
        ) {
            if (filteredPerformances.isEmpty()) {
                item {
                    Text(
                        text = stringResource(R.string.taste_empty),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 32.dp)
                    )
                }
            } else {
                items(sortedPerformances, key = { it.id }) { performance ->
                    TastePerformanceRow(
                        performance = performance,
                        onClick = { onPerformanceClick(performance.id) },
                        onRemoveClick = {
                            onRemoveFavorite(performance.id)
                            pendingRemovedId = performance.id
                        }
                    )
                }
            }
        }
        ClassicFyBottomBar(
            activeDestination = ClassicFyBottomDestination.TASTE,
            onSearchClick = onSearchTabClick,
            onTasteClick = onTasteTabClick,
            onMyClick = onMyTabClick
        )
    }

    if (showSortSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSortSheet = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp)
                    .padding(bottom = 32.dp)
            ) {
                Text(
                    text = stringResource(R.string.taste_sort_criterion),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f)
                )
                Spacer(Modifier.height(8.dp))
                listOf(
                    TasteSortCriterion.ADDED_DATE to R.string.taste_sort_added_date,
                    TasteSortCriterion.COMPOSER to R.string.taste_sort_composer,
                    TasteSortCriterion.WORK_TITLE to R.string.taste_sort_work
                ).forEach { (criterion, labelRes) ->
                    TasteSortOption(
                        label = stringResource(labelRes),
                        selected = sortCriterion == criterion,
                        onClick = {
                            if (sortCriterion != criterion) {
                                sortCriterion = criterion
                                sortDirection = if (criterion == TasteSortCriterion.ADDED_DATE) {
                                    TasteSortDirection.DESCENDING
                                } else TasteSortDirection.ASCENDING
                            }
                        }
                    )
                }
                Spacer(Modifier.height(24.dp))
                HorizontalDivider(
                    thickness = 1.dp,
                    color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                )
                Spacer(Modifier.height(24.dp))
                Text(
                    text = stringResource(R.string.taste_sort_direction),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.85f)
                )
                Spacer(Modifier.height(8.dp))
                val directions = if (sortCriterion == TasteSortCriterion.ADDED_DATE) {
                    listOf(
                        TasteSortDirection.DESCENDING to R.string.taste_sort_recent,
                        TasteSortDirection.ASCENDING to R.string.taste_sort_oldest
                    )
                } else {
                    listOf(
                        TasteSortDirection.ASCENDING to R.string.taste_sort_ascending,
                        TasteSortDirection.DESCENDING to R.string.taste_sort_descending
                    )
                }
                directions.forEach { (direction, labelRes) ->
                    TasteSortOption(
                        label = stringResource(labelRes),
                        selected = sortDirection == direction,
                        onClick = {
                            sortDirection = direction
                            showSortSheet = false
                        }
                    )
                }
            }
        }
    }

    if (pendingRemovedId != null) {
        FavoriteRemovalNotice(
            onUndo = undoRemoval,
            onConfirm = { pendingRemovedId = null }
        )
    }
}

@Composable
private fun TasteSortOption(label: String, selected: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .clickable(onClick = onClick),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyLarge,
            color = if (selected) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.weight(1f)
        )
        if (selected) {
            Text(text = "✓", color = MaterialTheme.colorScheme.primary)
        }
    }
}

@Composable
private fun TasteSearchField(query: String, onQueryChange: (String) -> Unit) {
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
                            text = stringResource(R.string.preference_search_placeholder),
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
private fun TastePerformanceRow(
    performance: PerformanceItem,
    onClick: () -> Unit,
    onRemoveClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 76.dp)
            .clickable(onClick = onClick),
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
        IconButton(onClick = onRemoveClick) {
            Icon(
                painter = painterResource(R.drawable.ic_favorite_filled),
                contentDescription = stringResource(R.string.performance_remove_favorite),
                tint = MaterialTheme.colorScheme.primary
            )
        }
        Icon(
            painter = painterResource(R.drawable.ic_chevron_right),
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier
                .padding(end = 4.dp)
                .size(20.dp)
        )
    }
    HorizontalDivider(
        thickness = 1.dp,
        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
    )
}
