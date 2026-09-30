package com.classicfy.app.ui.screen.signup

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsBottomHeight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyTheme
import com.classicfy.app.ui.validation.isMockIdTaken
import com.classicfy.app.ui.validation.MIN_PASSWORD_LENGTH

@Composable
fun SignUpScreen(
    onSignUpClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var id by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var nickname by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    val trimmedId = id.trim()
    val idTaken = trimmedId.isNotEmpty() && isMockIdTaken(trimmedId)
    val canSignUp = trimmedId.isNotEmpty() && !idTaken &&
        password.length >= MIN_PASSWORD_LENGTH && nickname.isNotBlank()
    val idStatus = when {
        trimmedId.isEmpty() -> null
        idTaken -> R.string.my_id_duplicate
        else -> R.string.my_id_available
    }
    val passwordStatus = when {
        password.isEmpty() -> null
        password.length < MIN_PASSWORD_LENGTH ->
            stringResource(R.string.password_change_min_length, MIN_PASSWORD_LENGTH)
        else -> null
    }
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        val logoSize = minOf(maxWidth * 0.30f, 112.dp)
        val headerBottomSpacing = minOf(maxHeight * 0.06f, 48.dp)
        val buttonTopSpacing = minOf(maxHeight * 0.08f, 64.dp)

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier
                    .widthIn(max = 440.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = stringResource(R.string.sign_up_title),
                    style = MaterialTheme.typography.headlineLarge,
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Image(
                    painter = painterResource(R.drawable.classicfy_logo),
                    contentDescription = stringResource(R.string.app_name),
                    modifier = Modifier
                        .width(logoSize)
                        .height(logoSize * 0.76f)
                        .clipToBounds(),
                    contentScale = ContentScale.FillWidth,
                    alignment = Alignment.TopCenter
                )
            }
            Spacer(Modifier.height(headerBottomSpacing))

            Column(
                modifier = Modifier
                    .widthIn(max = 440.dp)
                    .fillMaxWidth()
            ) {
                SignUpInput(
                    label = stringResource(R.string.login_id_label),
                    placeholder = stringResource(R.string.sign_up_id_placeholder),
                    value = id,
                    onValueChange = { id = it },
                    statusMessage = idStatus?.let { stringResource(it) },
                    keyboardOptions = KeyboardOptions(
                        autoCorrectEnabled = false,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { focusManager.moveFocus(FocusDirection.Down) }
                    )
                )
                Spacer(Modifier.height(40.dp))
                SignUpInput(
                    label = stringResource(R.string.login_password_label),
                    placeholder = stringResource(R.string.sign_up_password_placeholder),
                    value = password,
                    onValueChange = { password = it },
                    statusMessage = passwordStatus,
                    visualTransformation = if (passwordVisible) {
                        VisualTransformation.None
                    } else {
                        PasswordVisualTransformation()
                    },
                    keyboardOptions = KeyboardOptions(
                        autoCorrectEnabled = false,
                        keyboardType = KeyboardType.Password,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { focusManager.moveFocus(FocusDirection.Down) }
                    ),
                    trailingIcon = {
                        IconButton(onClick = { passwordVisible = !passwordVisible }) {
                            Icon(
                                painter = painterResource(
                                    if (passwordVisible) R.drawable.ic_password_hidden
                                    else R.drawable.ic_password_visible
                                ),
                                contentDescription = stringResource(
                                    if (passwordVisible) R.string.login_hide_password
                                    else R.string.login_show_password
                                ),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                )
                Spacer(Modifier.height(40.dp))
                SignUpInput(
                    label = stringResource(R.string.sign_up_nickname_label),
                    placeholder = stringResource(R.string.sign_up_nickname_placeholder),
                    value = nickname,
                    onValueChange = { nickname = it },
                    statusMessage = null,
                    keyboardOptions = KeyboardOptions(
                        autoCorrectEnabled = false,
                        imeAction = ImeAction.Done
                    ),
                    keyboardActions = KeyboardActions(
                        onDone = {
                            keyboardController?.hide()
                            focusManager.clearFocus()
                        }
                    )
                )
                Spacer(Modifier.height(buttonTopSpacing))
                Button(
                    onClick = {
                        keyboardController?.hide()
                        focusManager.clearFocus()
                        onSignUpClick(nickname.trim())
                    },
                    enabled = canSignUp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 60.dp),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Text(
                        text = stringResource(R.string.sign_up_button),
                        style = MaterialTheme.typography.titleMedium,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(Modifier.height(32.dp))
            }
            // Reserve the keyboard's height as scrollable content after the form.
            Spacer(
                Modifier.windowInsetsBottomHeight(
                    WindowInsets.ime.exclude(WindowInsets.navigationBars)
                )
            )
        }
    }
}

@Composable
private fun SignUpInput(
    label: String,
    placeholder: String,
    value: String,
    onValueChange: (String) -> Unit,
    statusMessage: String?,
    keyboardOptions: KeyboardOptions,
    keyboardActions: KeyboardActions,
    visualTransformation: VisualTransformation = VisualTransformation.None,
    trailingIcon: @Composable (() -> Unit)? = null
) {
    var focused by remember { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            text = label,
            color = MaterialTheme.colorScheme.onBackground,
            style = MaterialTheme.typography.bodyLarge,
            fontSize = 18.sp,
            fontWeight = FontWeight.Medium
        )
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                BasicTextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 56.dp)
                        .onFocusChanged { focused = it.isFocused }
                        .semantics { contentDescription = label },
                    singleLine = true,
                    visualTransformation = visualTransformation,
                    keyboardOptions = keyboardOptions,
                    keyboardActions = keyboardActions,
                    textStyle = MaterialTheme.typography.bodyLarge.copy(
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 17.sp
                    ),
                    cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                    decorationBox = { innerTextField ->
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.CenterStart
                        ) {
                            if (value.isEmpty()) {
                                Text(
                                    text = placeholder,
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontSize = 17.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            innerTextField()
                        }
                    }
                )
                trailingIcon?.invoke()
            }
            HorizontalDivider(
                color = if (focused) MaterialTheme.colorScheme.primary
                else MaterialTheme.colorScheme.outline,
                thickness = if (focused) 2.dp else 1.dp
            )
            if (statusMessage != null) {
                Text(
                    text = statusMessage,
                    style = MaterialTheme.typography.bodySmall,
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(start = 2.dp, top = 4.dp)
                )
            }
        }
    }
}

@Preview(name = "Small phone", widthDp = 320, heightDp = 640)
@Preview(name = "Large phone", widthDp = 412, heightDp = 915)
@Composable
private fun SignUpScreenPreview() {
    ClassicFyTheme {
        SignUpScreen(onSignUpClick = {})
    }
}
