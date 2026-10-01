package com.classicfy.app.ui.screen.signup

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyPrimaryButton
import com.classicfy.app.ui.theme.ClassicFyTheme

@Composable
fun SignUpCompleteScreen(
    nickname: String,
    onStartClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 440.dp)
                .fillMaxWidth()
                .padding(start = 24.dp, end = 24.dp, bottom = 120.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(R.drawable.classicfy_logo),
                contentDescription = stringResource(R.string.app_name),
                modifier = Modifier.size(105.dp),
                contentScale = ContentScale.Fit
            )
            Text(
                text = stringResource(R.string.sign_up_complete_title),
                style = MaterialTheme.typography.headlineLarge,
                fontSize = 44.sp,
                lineHeight = 52.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(28.dp))
            Text(
                text = stringResource(R.string.sign_up_complete_message) + "\n" +
                    stringResource(R.string.sign_up_complete_nickname_message, nickname),
                style = MaterialTheme.typography.bodyLarge,
                fontSize = 18.sp,
                lineHeight = 26.sp,
                color = MaterialTheme.colorScheme.onBackground,
                textAlign = TextAlign.Center
            )
            Spacer(Modifier.height(56.dp))
            ClassicFyPrimaryButton(
                onClick = onStartClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text(
                    text = stringResource(R.string.sign_up_complete_start),
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }
    }
}

@Preview(name = "Sign Up Complete", widthDp = 360, heightDp = 800)
@Composable
private fun SignUpCompleteScreenPreview() {
    ClassicFyTheme {
        SignUpCompleteScreen(nickname = "민수", onStartClick = {})
    }
}
