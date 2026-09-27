package com.classicfy.app.ui.screen.splash

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyTheme

@Composable
fun SplashScreen(modifier: Modifier = Modifier) {
    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        // The artwork fills about 58% of the PNG width, so 78% gives a visible logo near 45%.
        val logoSize = minOf(maxWidth * 0.78f, maxHeight * 0.75f)

        Image(
            painter = painterResource(R.drawable.classicfy_logo),
            contentDescription = "ClassicFy logo",
            modifier = Modifier.size(logoSize),
            contentScale = ContentScale.Fit
        )
    }
}

@Preview(name = "Small phone", widthDp = 320, heightDp = 640)
@Preview(name = "Large phone", widthDp = 412, heightDp = 915)
@Composable
private fun SplashScreenPreview() {
    ClassicFyTheme {
        SplashScreen()
    }
}
