package com.classicfy.app.ui.screen.my

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.classicfy.app.R
import com.classicfy.app.ui.theme.ClassicFyPrimaryButton
import com.classicfy.app.ui.navigation.ClassicFyBottomBar
import com.classicfy.app.ui.navigation.ClassicFyBottomDestination
import com.classicfy.app.ui.validation.isMockIdTaken

private enum class IdValidation { DEFAULT, DUPLICATE, AVAILABLE }

@Composable
fun MyPageScreen(
    savedNickname: String,
    savedId: String,
    onSave: (String, String) -> Unit,
    onLogoutClick: () -> Unit,
    onPasswordChangeClick: () -> Unit,
    onSearchTabClick: () -> Unit,
    onTasteTabClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    var nickname by rememberSaveable(savedNickname) { mutableStateOf(savedNickname) }
    var id by rememberSaveable(savedId) { mutableStateOf(savedId) }
    val trimmedId = id.trim()
    val idValidation = when {
        trimmedId.isEmpty() || trimmedId.equals(savedId, ignoreCase = true) ->
            IdValidation.DEFAULT
        isMockIdTaken(trimmedId) ->
            IdValidation.DUPLICATE
        else -> IdValidation.AVAILABLE
    }
    val canSave = nickname.isNotBlank() && trimmedId.isNotEmpty() &&
        idValidation != IdValidation.DUPLICATE
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .imePadding()
    ) {
        Column(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
        ) {
            Spacer(Modifier.height(20.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = stringResource(R.string.my_page_title),
                    style = MaterialTheme.typography.headlineLarge,
                    color = MaterialTheme.colorScheme.onBackground,
                    modifier = Modifier.padding(start = 24.dp)
                )
                TextButton(
                    onClick = onLogoutClick,
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .padding(end = 16.dp)
                ) {
                    Text(
                        text = stringResource(R.string.my_logout),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
            Spacer(Modifier.height(40.dp))
            Column(modifier = Modifier.padding(horizontal = 24.dp)) {
                Text(
                    text = stringResource(R.string.my_nickname_label),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(12.dp))
                ProfileField(
                    value = nickname,
                    onValueChange = { nickname = it },
                    imeAction = ImeAction.Next,
                    keyboardActions = KeyboardActions(
                        onNext = { focusManager.moveFocus(FocusDirection.Down) }
                    )
                )
                Spacer(Modifier.height(32.dp))
                Text(
                    text = stringResource(R.string.my_id_label),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(Modifier.height(12.dp))
                ProfileField(
                    value = id,
                    onValueChange = { id = it },
                    imeAction = ImeAction.Done,
                    keyboardActions = KeyboardActions(
                        onDone = {
                            focusManager.clearFocus()
                            keyboardController?.hide()
                        }
                    )
                )
                if (idValidation != IdValidation.DEFAULT) {
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = stringResource(
                            if (idValidation == IdValidation.DUPLICATE) {
                                R.string.my_id_duplicate
                            } else {
                                R.string.my_id_available
                            }
                        ),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(start = 16.dp)
                    )
                }
                Spacer(Modifier.height(56.dp))
                ClassicFyPrimaryButton(
                    onClick = {
                        focusManager.clearFocus()
                        keyboardController?.hide()
                        onSave(nickname.trim(), trimmedId)
                    },
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
                Spacer(Modifier.height(8.dp))
                TextButton(
                    onClick = onPasswordChangeClick,
                    modifier = Modifier
                        .align(Alignment.End)
                        .heightIn(min = 48.dp),
                    contentPadding = PaddingValues(horizontal = 12.dp)
                ) {
                    Text(
                        text = stringResource(R.string.my_change_password),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                }
                Spacer(Modifier.height(24.dp))
            }
        }
        ClassicFyBottomBar(
            activeDestination = ClassicFyBottomDestination.MY,
            onSearchClick = onSearchTabClick,
            onTasteClick = onTasteTabClick,
            onMyClick = {}
        )
    }
}

@Composable
private fun ProfileField(
    value: String,
    onValueChange: (String) -> Unit,
    imeAction: ImeAction,
    keyboardActions: KeyboardActions
) {
    BasicTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(14.dp)),
        singleLine = true,
        keyboardOptions = KeyboardOptions(imeAction = imeAction),
        keyboardActions = keyboardActions,
        textStyle = MaterialTheme.typography.bodyLarge.copy(
            color = MaterialTheme.colorScheme.onSurface
        ),
        cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
        decorationBox = { innerTextField ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp),
                contentAlignment = Alignment.CenterStart
            ) {
                innerTextField()
            }
        }
    )
}
