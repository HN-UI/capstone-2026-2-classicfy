package com.classicfy.app.ui.screen.preference

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyPrimaryButton

private data class PreferenceReportItem(
    val feature: String,
    val description: String
)

@Composable
fun PreferenceResultScreen(
    onSearchClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val heading = stringResource(R.string.preference_result_heading)
    val highlightedText = stringResource(R.string.preference_result_heading_highlight)
    val highlightStart = heading.indexOf(highlightedText)
    val reportItems = listOf(
        PreferenceReportItem("Tempo", stringResource(R.string.preference_result_tempo)),
        PreferenceReportItem("Rubato", stringResource(R.string.preference_result_rubato)),
        PreferenceReportItem("Dynamics", stringResource(R.string.preference_result_dynamics)),
        PreferenceReportItem("Articulation", stringResource(R.string.preference_result_articulation)),
        PreferenceReportItem("Pedaling", stringResource(R.string.preference_result_pedaling))
    )

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(start = 24.dp, end = 24.dp, top = 20.dp, bottom = 24.dp)
        ) {
            Text(
                text = stringResource(R.string.preference_title),
                style = MaterialTheme.typography.headlineLarge,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(28.dp))
            Text(
                text = buildAnnotatedString {
                    append(heading)
                    if (highlightStart >= 0) {
                        addStyle(
                            SpanStyle(color = MaterialTheme.colorScheme.primary),
                            highlightStart,
                            highlightStart + highlightedText.length
                        )
                    }
                },
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = stringResource(R.string.preference_result_supporting_text),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(42.dp))
            Text(
                text = stringResource(R.string.preference_result_report_title),
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onBackground
            )
            Spacer(Modifier.height(12.dp))
            reportItems.forEach { item ->
                PreferenceReportRow(item)
            }
        }

        ClassicFyPrimaryButton(
            onClick = onSearchClick,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 16.dp)
                .height(58.dp)
        ) {
            Text(
                text = stringResource(R.string.preference_result_search_button),
                style = MaterialTheme.typography.labelLarge
            )
        }
    }
}

@Composable
private fun PreferenceReportRow(item: PreferenceReportItem) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 60.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = item.feature,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = item.description,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurface,
            textAlign = TextAlign.End
        )
    }
    HorizontalDivider(
        thickness = 1.dp,
        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.25f)
    )
}
