package com.classicfy.app.ui.screen.login

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyPrimaryButton
import com.classicfy.app.ui.theme.ClassicFyTheme

private const val MOCK_LOGIN_ID = "classicfy"
internal const val MOCK_LOGIN_PASSWORD = "1234"

@Composable
fun LoginScreen(
    onLoginClick: () -> Unit,
    modifier: Modifier = Modifier,
    onSignUpClick: () -> Unit = {},
    validPassword: String = MOCK_LOGIN_PASSWORD
) {
    var id by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var showIdRequired by remember { mutableStateOf(false) }
    var showPasswordRequired by remember { mutableStateOf(false) }
    var showInvalidCredentials by remember { mutableStateOf(false) }
    var passwordVisible by remember { mutableStateOf(false) }
    var passwordFocused by remember { mutableStateOf(false) }
    var viewportHeight by remember { mutableIntStateOf(0) }
    val passwordActionsRequester = remember { BringIntoViewRequester() }
    val imeBottom = WindowInsets.ime.getBottom(LocalDensity.current)
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val submit = {
        keyboardController?.hide()
        focusManager.clearFocus()
        showIdRequired = id.isEmpty()
        showPasswordRequired = password.isEmpty()
        showInvalidCredentials = false
        if (!showIdRequired && !showPasswordRequired) {
            if (id == MOCK_LOGIN_ID && password == validPassword) {
                onLoginClick()
            } else {
                showInvalidCredentials = true
            }
        }
    }

    // Retry after IME layout changes, rather than only at the initial focus event.
    LaunchedEffect(passwordFocused, imeBottom, viewportHeight) {
        if (passwordFocused && imeBottom > 0 && viewportHeight > 0) {
            passwordActionsRequester.bringIntoView()
        }
    }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        val logoSize = minOf(maxWidth * 0.70f, maxHeight * 0.325f, 275.dp) * 0.84f
        val topSpacing = maxHeight * 0.12f
        // Account for the cropped logo box while keeping the form close to the artwork.
        val logoBottomSpacing = 16.dp + logoSize * 0.08f

        Column(
            modifier = Modifier
                .fillMaxSize()
                // Scaffold supplies system-bar padding; apply only the remaining IME area.
                .windowInsetsPadding(WindowInsets.ime.exclude(WindowInsets.navigationBars))
                .onSizeChanged { viewportHeight = it.height }
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(topSpacing))
            Image(
                painter = painterResource(R.drawable.classicfy_logo),
                contentDescription = stringResource(R.string.app_name),
                // Preserve the artwork's scale, removing only the PNG's transparent bottom.
                modifier = Modifier
                    .width(logoSize)
                    .height(logoSize * 0.76f)
                    .clipToBounds(),
                contentScale = ContentScale.FillWidth,
                alignment = Alignment.TopCenter
            )
            Spacer(Modifier.height(logoBottomSpacing))

            Column(
                modifier = Modifier
                    .widthIn(max = 420.dp)
                    .fillMaxWidth()
            ) {
                LoginInput(
                    label = stringResource(R.string.login_id_label),
                    placeholder = stringResource(R.string.login_id_placeholder),
                    value = id,
                    onValueChange = {
                        id = it
                        if (it.isNotEmpty()) showIdRequired = false
                        showInvalidCredentials = false
                    },
                    errorText = if (showIdRequired) {
                        stringResource(R.string.login_id_required)
                    } else null,
                    keyboardOptions = KeyboardOptions(
                        autoCorrectEnabled = false,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(
                        onNext = { focusManager.moveFocus(FocusDirection.Down) }
                    )
                )
                Spacer(Modifier.height(32.dp))
                Column(
                    modifier = Modifier.bringIntoViewRequester(passwordActionsRequester)
                ) {
                    LoginInput(
                        label = stringResource(R.string.login_password_label),
                        placeholder = stringResource(R.string.login_password_placeholder),
                        value = password,
                        onValueChange = {
                            password = it
                            if (it.isNotEmpty()) showPasswordRequired = false
                            showInvalidCredentials = false
                        },
                        errorText = if (showPasswordRequired) {
                            stringResource(R.string.login_password_required)
                        } else null,
                        visualTransformation = if (passwordVisible) {
                            VisualTransformation.None
                        } else {
                            PasswordVisualTransformation()
                        },
                        keyboardOptions = KeyboardOptions(
                            autoCorrectEnabled = false,
                            keyboardType = KeyboardType.Password,
                            imeAction = ImeAction.Done
                        ),
                        keyboardActions = KeyboardActions(onDone = { submit() }),
                        modifier = Modifier.onFocusChanged { passwordFocused = it.isFocused },
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
                                    modifier = Modifier.size(24.dp),
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    )
                    if (showInvalidCredentials) {
                        Text(
                            text = stringResource(R.string.login_invalid_credentials),
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.labelSmall,
                            textAlign = TextAlign.Center
                        )
                    }
                    Spacer(Modifier.height(44.dp))
                    ClassicFyPrimaryButton(
                        onClick = submit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(58.dp),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.login_button),
                            style = MaterialTheme.typography.labelLarge
                        )
                    }
                }
                Spacer(Modifier.height(14.dp))
                TextButton(
                    onClick = onSignUpClick,
                    modifier = Modifier.align(Alignment.End),
                    contentPadding = PaddingValues(0.dp)
                ) {
                    Text(
                        text = stringResource(R.string.login_sign_up),
                        color = MaterialTheme.colorScheme.onBackground,
                        style = MaterialTheme.typography.bodyLarge,
                        textDecoration = TextDecoration.Underline
                    )
                }
                Spacer(Modifier.height(32.dp))
            }
        }
    }
}

@Composable
private fun LoginInput(
    label: String,
    placeholder: String,
    value: String,
    onValueChange: (String) -> Unit,
    errorText: String?,
    keyboardOptions: KeyboardOptions,
    keyboardActions: KeyboardActions,
    modifier: Modifier = Modifier,
    visualTransformation: VisualTransformation = VisualTransformation.None,
    trailingIcon: @Composable (() -> Unit)? = null
) {
    Column {
        Text(
            text = label,
            color = MaterialTheme.colorScheme.onBackground,
            style = MaterialTheme.typography.titleMedium
        )
        Spacer(Modifier.height(8.dp))
        TextField(
            value = value,
            onValueChange = onValueChange,
            modifier = modifier
                .fillMaxWidth()
                .height(58.dp)
                .semantics { contentDescription = label },
            placeholder = { Text(placeholder, style = MaterialTheme.typography.bodyLarge) },
            textStyle = MaterialTheme.typography.bodyLarge,
            singleLine = true,
            shape = RoundedCornerShape(14.dp),
            visualTransformation = visualTransformation,
            trailingIcon = trailingIcon,
            keyboardOptions = keyboardOptions,
            keyboardActions = keyboardActions,
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color(0xFF464646),
                unfocusedContainerColor = Color(0xFF464646),
                focusedTextColor = MaterialTheme.colorScheme.onSurface,
                unfocusedTextColor = MaterialTheme.colorScheme.onSurface,
                focusedPlaceholderColor = Color(0xFFD9D9D9),
                unfocusedPlaceholderColor = Color(0xFFD9D9D9),
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent
            )
        )
        if (errorText != null) {
            Text(
                text = errorText,
                modifier = Modifier.padding(start = 12.dp, top = 4.dp),
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.labelSmall
            )
        }
    }
}

@Preview(name = "Small phone", widthDp = 320, heightDp = 640)
@Preview(name = "Large phone", widthDp = 412, heightDp = 915)
@Composable
private fun LoginScreenPreview() {
    ClassicFyTheme {
        LoginScreen(onLoginClick = {})
    }
}
