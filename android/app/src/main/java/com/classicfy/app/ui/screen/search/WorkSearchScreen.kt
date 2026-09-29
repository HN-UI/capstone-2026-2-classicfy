package com.classicfy.app.ui.screen.search

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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination

@Composable
fun WorkSearchScreen(
    onWorkClick: (String) -> Unit,
    onSearchTabClick: () -> Unit,
    onTasteTabClick: () -> Unit,
    onMyTabClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var query by rememberSaveable(stateSaver = TextFieldValue.Saver) {
        mutableStateOf(TextFieldValue(""))
    }
    var submittedQuery by rememberSaveable { mutableStateOf<String?>(null) }
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val submitSearch = {
        query = query.copy(selection = TextRange(query.text.length))
        submittedQuery = query.text.trim().takeIf { it.isNotEmpty() }
    }
    val searchTerm = (submittedQuery ?: query.text).trim()
    val matchingWorks = remember(searchTerm) {
        mockWorks.filter { work ->
            work.composer.contains(searchTerm, ignoreCase = true) ||
                work.title.contains(searchTerm, ignoreCase = true)
        }
    }
    val isInitial = query.text.isBlank() && submittedQuery == null
    val isTyping = !query.text.isBlank() && submittedQuery == null
    val hasSubmittedResults = submittedQuery != null && matchingWorks.isNotEmpty()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            // The search route extends into the navigation-bar area; IME padding
            // leaves the field and results usable while the keyboard is open.
            .imePadding()
    ) {
        if (hasSubmittedResults) {
            IconButton(
                onClick = {
                    query = TextFieldValue("")
                    submittedQuery = null
                    focusManager.clearFocus()
                    keyboardController?.hide()
                },
                modifier = Modifier.padding(start = 12.dp, top = 8.dp)
            ) {
                Icon(
                    painter = painterResource(R.drawable.ic_arrow_back),
                    contentDescription = stringResource(R.string.work_search_back_to_initial),
                    tint = MaterialTheme.colorScheme.onBackground
                )
            }
            Spacer(Modifier.height(12.dp))
        } else {
            Spacer(Modifier.height(20.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = stringResource(R.string.work_search_title),
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground,
                    modifier = Modifier.padding(start = 24.dp)
                )
            }
            Spacer(Modifier.height(20.dp))
        }
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp)
        ) {
            WorkSearchField(
                query = query,
                onQueryChange = {
                    if (it.text != query.text) submittedQuery = null
                    query = it
                },
                onSearch = submitSearch,
                modifier = Modifier.fillMaxWidth()
            )
            if (!isInitial) {
                Spacer(Modifier.height(28.dp))
                Text(
                    text = stringResource(R.string.work_search_results_title),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onBackground
                )
            }
        }

        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 24.dp, vertical = 16.dp)
        ) {
            if (isTyping) {
                if (matchingWorks.isEmpty()) {
                    item {
                        Text(
                            text = stringResource(R.string.work_search_no_suggestions),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 24.dp)
                        )
                    }
                } else {
                    items(matchingWorks.take(3), key = { it.id }) { work ->
                        WorkSearchRow(
                            work = work,
                            isSuggestion = true,
                            onClick = {
                                query = TextFieldValue(
                                    text = work.title,
                                    selection = TextRange(work.title.length)
                                )
                                submittedQuery = work.title
                            }
                        )
                    }
                }
            } else if (hasSubmittedResults) {
                items(matchingWorks, key = { it.id }) { work ->
                    WorkSearchRow(
                        work = work,
                        isSuggestion = false,
                        onClick = {
                            query = query.copy(selection = TextRange(query.text.length))
                            onWorkClick(work.id)
                        }
                    )
                }
            } else if (submittedQuery != null) {
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 72.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.work_search_no_results_title),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            text = stringResource(R.string.work_search_no_results_body),
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            textAlign = TextAlign.Center
                        )
                    }
                }
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
private fun WorkSearchField(
    query: TextFieldValue,
    onQueryChange: (TextFieldValue) -> Unit,
    onSearch: () -> Unit,
    modifier: Modifier
) {
    val placeholder = stringResource(R.string.work_search_placeholder)

    BasicTextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = modifier
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
                    if (query.text.isEmpty()) {
                        Text(
                            text = placeholder,
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
private fun WorkSearchRow(
    work: Work,
    isSuggestion: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clickable(onClick = onClick),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            painter = painterResource(
                if (isSuggestion) R.drawable.ic_search else R.drawable.ic_music_note
            ),
            contentDescription = null,
            tint = if (isSuggestion) MaterialTheme.colorScheme.onSurfaceVariant
            else MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.size(22.dp)
        )
        Text(
            text = "${work.composer}: ${work.title}",
            style = if (isSuggestion) MaterialTheme.typography.bodyLarge
            else MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f)
        )
        if (isSuggestion) {
            Text(
                text = "↖",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f),
                textAlign = TextAlign.Center,
                modifier = Modifier.width(20.dp)
            )
        } else {
            Icon(
                painter = painterResource(R.drawable.ic_chevron_right),
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp)
            )
        }
    }
    HorizontalDivider(
        thickness = 1.dp,
        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
    )
}
