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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.classicfy.app.R
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination
import com.classicfy.app.ui.screen.search.Work
import com.classicfy.app.ui.screen.search.mockWorks

private data class PerformanceItem(
    val id: String,
    val composer: String,
    val work: String,
    val performer: String,
    val isFavorite: Boolean = false
)

private val mockPerformersByWork = mapOf(
    "bach_goldberg" to listOf("글렌 굴드", "안드라스 시프", "머리 페라이어", "앙겔라 휴이트"),
    "bach_cello_suites" to listOf("요요 마", "파블로 카잘스", "미샤 마이스키", "장 기엔 케라스"),
    "beethoven_moonlight" to listOf("빌헬름 켐프", "알프레트 브렌델", "다니엘 바렌보임", "미츠코 우치다"),
    "beethoven_symphony_5" to listOf("카를로스 클라이버", "헤르베르트 폰 카라얀", "레너드 번스타인", "빌헬름 푸르트벵글러"),
    "chopin_nocturne" to listOf("아르투르 루빈스타인", "마리아 주앙 피르스", "클라우디오 아라우", "조성진"),
    "mozart_requiem" to listOf("카를 뵘", "니콜라우스 아르농쿠르", "존 엘리엇 가디너", "헤르베르트 폰 카라얀"),
    "debussy_clair_de_lune" to listOf("조성진", "크리스티안 지메르만", "마르타 아르헤리치", "알리스 사라 오트"),
    "vivaldi_four_seasons" to listOf("안네 소피 무터", "나이절 케네디", "야니네 얀선", "레이첼 포저"),
    "rachmaninoff_piano_concerto_2" to listOf("블라디미르 아슈케나지", "마르타 아르헤리치", "유자 왕", "스비아토슬라프 리히테르"),
    "tchaikovsky_swan_lake" to listOf("발레리 게르기예프", "헤르베르트 폰 카라얀", "구스타보 두다멜", "예브게니 스베틀라노프")
)

private fun mockPerformancesFor(work: Work): List<PerformanceItem> =
    requireNotNull(mockPerformersByWork[work.id]).mapIndexed { index, performer ->
        PerformanceItem(
            id = "${work.id}_$index",
            composer = work.composer,
            work = work.title,
            performer = performer
        )
    }

@Composable
fun PerformanceSelectionScreen(
    workId: String,
    onBackClick: () -> Unit,
    onPerformanceClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var selectedWorkId by rememberSaveable(workId) { mutableStateOf(workId) }
    val selectedWork = remember(selectedWorkId) {
        requireNotNull(mockWorks.find { it.id == selectedWorkId })
    }
    val performances = remember(selectedWorkId) { mockPerformancesFor(selectedWork) }
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
        quickQuery = ""
        focusManager.clearFocus()
        keyboardController?.hide()
    }
    var favoriteIds by rememberSaveable(workId) {
        mutableStateOf(arrayListOf<String>())
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
        if (suggestions.isNotEmpty()) {
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
                suggestions.forEach { work ->
                    QuickWorkSuggestion(work = work, onClick = { selectWork(work) })
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
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
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
                    performance = performance.copy(isFavorite = performance.id in favoriteIds),
                    onFavoriteClick = {
                        favoriteIds = ArrayList(favoriteIds).apply {
                            if (!remove(performance.id)) add(performance.id)
                        }
                    },
                    onClick = onPerformanceClick
                )
            }
        }
        ClassicFyBottomBar(
            activeDestination = ClassicFyBottomDestination.SEARCH,
            onSearchClick = onBackClick
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
    onFavoriteClick: () -> Unit,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 72.dp)
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
        IconButton(onClick = onFavoriteClick) {
            Icon(
                painter = painterResource(
                    if (performance.isFavorite) R.drawable.ic_favorite_filled
                    else R.drawable.ic_taste
                ),
                contentDescription = stringResource(
                    if (performance.isFavorite) R.string.performance_remove_favorite
                    else R.string.performance_add_favorite
                ),
                tint = if (performance.isFavorite) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Icon(
            painter = painterResource(R.drawable.ic_chevron_right),
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurface,
            modifier = Modifier.size(20.dp)
        )
    }
    HorizontalDivider(
        thickness = 1.dp,
        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
    )
}
