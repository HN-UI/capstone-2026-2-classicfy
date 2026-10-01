package com.classicfy.app.ui.screen.my

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyPrimaryButton
import com.classicfy.app.ui.theme.ClassicFyTheme
import com.classicfy.app.ui.validation.MIN_PASSWORD_LENGTH

@Composable
fun PasswordChangeScreen(
    onBackClick: () -> Unit,
    onSaveClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    var newPassword by remember { mutableStateOf("") }
    var confirmation by remember { mutableStateOf("") }
    var newPasswordVisible by remember { mutableStateOf(false) }
    var confirmationVisible by remember { mutableStateOf(false) }
    val passwordsMatch = newPassword == confirmation
    val showMatchStatus = newPassword.isNotEmpty() && confirmation.isNotEmpty()
    val canSave = showMatchStatus && passwordsMatch &&
        newPassword.length >= MIN_PASSWORD_LENGTH
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .imePadding()
            .verticalScroll(rememberScrollState())
    ) {
        IconButton(
            onClick = onBackClick,
            modifier = Modifier.padding(start = 12.dp, top = 8.dp)
        ) {
            Icon(
                painter = painterResource(R.drawable.ic_arrow_back),
                contentDescription = stringResource(R.string.password_change_back),
                tint = MaterialTheme.colorScheme.onBackground
            )
        }
        Spacer(Modifier.height(8.dp))
        Column(
            modifier = Modifier
                .widthIn(max = 440.dp)
                .fillMaxWidth()
                .align(Alignment.CenterHorizontally)
                .padding(horizontal = 24.dp)
        ) {
            PasswordChangeField(
                label = stringResource(R.string.password_change_new_label),
                placeholder = stringResource(R.string.password_change_new_placeholder),
                value = newPassword,
                onValueChange = { newPassword = it },
                visible = newPasswordVisible,
                onVisibilityChange = { newPasswordVisible = !newPasswordVisible },
                imeAction = ImeAction.Next,
                keyboardActions = KeyboardActions(
                    onNext = { focusManager.moveFocus(androidx.compose.ui.focus.FocusDirection.Down) }
                )
            )
            if (newPassword.isNotEmpty() && newPassword.length < MIN_PASSWORD_LENGTH) {
                Text(
                    text = stringResource(
                        R.string.password_change_min_length,
                        MIN_PASSWORD_LENGTH
                    ),
                    color = MaterialTheme.colorScheme.primary,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(start = 12.dp, top = 4.dp)
                )
            }
            Spacer(Modifier.height(24.dp))
            PasswordChangeField(
                label = stringResource(R.string.password_change_confirm_label),
                placeholder = stringResource(R.string.password_change_confirm_placeholder),
                value = confirmation,
                onValueChange = { confirmation = it },
                visible = confirmationVisible,
                onVisibilityChange = { confirmationVisible = !confirmationVisible },
                imeAction = ImeAction.Done,
                keyboardActions = KeyboardActions(
                    onDone = {
                        keyboardController?.hide()
                        focusManager.clearFocus()
                    }
                )
            )
            Box(modifier = Modifier.height(24.dp)) {
                if (showMatchStatus) {
                    Text(
                        text = stringResource(
                            if (passwordsMatch) R.string.password_change_match
                            else R.string.password_change_mismatch
                        ),
                        color = MaterialTheme.colorScheme.primary,
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(start = 12.dp, top = 4.dp)
                    )
                }
            }
            Spacer(Modifier.height(32.dp))
            ClassicFyPrimaryButton(
                onClick = { onSaveClick(newPassword) },
                enabled = canSave,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text(
                    text = stringResource(R.string.my_save),
                    style = MaterialTheme.typography.labelLarge
                )
            }
            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
private fun PasswordChangeField(
    label: String,
    placeholder: String,
    value: String,
    onValueChange: (String) -> Unit,
    visible: Boolean,
    onVisibilityChange: () -> Unit,
    imeAction: ImeAction,
    keyboardActions: KeyboardActions
) {
    Text(
        text = label,
        style = MaterialTheme.typography.titleMedium,
        color = MaterialTheme.colorScheme.onBackground,
        modifier = Modifier.padding(start = 12.dp)
    )
    Spacer(Modifier.height(4.dp))
    TextField(
        value = value,
        onValueChange = onValueChange,
        modifier = Modifier
            .fillMaxWidth()
            .height(58.dp)
            .semantics { contentDescription = label },
        placeholder = { Text(placeholder, style = MaterialTheme.typography.bodyLarge) },
        textStyle = MaterialTheme.typography.bodyLarge,
        singleLine = true,
        shape = RoundedCornerShape(14.dp),
        visualTransformation = if (visible) VisualTransformation.None
        else PasswordVisualTransformation(),
        trailingIcon = {
            IconButton(onClick = onVisibilityChange) {
                Icon(
                    painter = painterResource(
                        if (visible) R.drawable.ic_password_hidden
                        else R.drawable.ic_password_visible
                    ),
                    contentDescription = stringResource(
                        if (visible) R.string.login_hide_password
                        else R.string.login_show_password
                    ),
                    modifier = Modifier.size(24.dp),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        keyboardOptions = KeyboardOptions(
            autoCorrectEnabled = false,
            keyboardType = KeyboardType.Password,
            imeAction = imeAction
        ),
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
}

@Preview(name = "Password Change", widthDp = 360, heightDp = 800)
@Composable
private fun PasswordChangeScreenPreview() {
    ClassicFyTheme {
        PasswordChangeScreen(onBackClick = {}, onSaveClick = {})
    }
}
