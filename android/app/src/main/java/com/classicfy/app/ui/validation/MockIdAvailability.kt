package com.classicfy.app.ui.validation

private val existingMockIds = setOf("admin", "classicfy", "user123")

internal fun isMockIdTaken(id: String): Boolean =
    existingMockIds.any { it.equals(id.trim(), ignoreCase = true) }
