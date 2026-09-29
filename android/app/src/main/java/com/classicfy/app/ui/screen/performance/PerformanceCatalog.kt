package com.classicfy.app.ui.screen.performance

import com.classicfy.app.ui.screen.search.Work
import com.classicfy.app.ui.screen.search.mockWorks

internal data class PerformanceItem(
    val id: String,
    val workId: String,
    val composer: String,
    val work: String,
    val performer: String,
    val rank: Int,
    val shortReason: String,
    val detailedReason: String
)

private data class MockReason(val short: String, val detailed: String)

private val mockReasons = listOf(
    MockReason(
        "안정적인 템포와 또렷한 프레이징이 선택한 취향과 잘 맞아요.",
        "일정한 템포를 유지하면서 선율의 흐름을 또렷하게 들려주는 연주예요. " +
            "강약의 변화가 자연스럽고 각 구절의 연결이 분명해, 선택한 연주에서 드러난 취향과 비슷한 인상을 줍니다."
    ),
    MockReason(
        "자유로운 호흡과 섬세한 강약 변화가 돋보여요.",
        "구절마다 호흡을 유연하게 조절해 음악의 흐름을 살린 연주예요. " +
            "섬세한 강약 변화가 선율에 표정을 더하며, 익숙한 작품도 새롭게 들을 수 있게 해줍니다."
    ),
    MockReason(
        "선명한 아티큘레이션으로 작품의 구조가 잘 드러나요.",
        "음 하나하나의 시작과 끝을 분명하게 표현해 작품의 구조를 따라가기 좋아요. " +
            "빠른 부분에서도 소리가 흐려지지 않고, 중요한 선율이 자연스럽게 앞으로 나옵니다."
    ),
    MockReason(
        "절제된 표현 속에서 부드러운 음색을 느낄 수 있어요.",
        "과장된 효과보다 균형 잡힌 음색과 차분한 흐름에 집중한 연주예요. " +
            "잔향과 악기 사이의 균형을 절제해, 작품의 분위기를 편안하게 감상할 수 있습니다."
    )
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

private val mockPerformances = mockWorks.flatMap { work ->
    requireNotNull(mockPerformersByWork[work.id]).mapIndexed { index, performer ->
        val reason = mockReasons[index]
        PerformanceItem(
            id = "${work.id}_$index",
            workId = work.id,
            composer = work.composer,
            work = work.title,
            performer = performer,
            rank = index + 1,
            shortReason = reason.short,
            detailedReason = reason.detailed
        )
    }
}

// A small mock taste profile until onboarding selections are connected to app data.
internal val initialTastePerformanceIds = setOf(
    "bach_goldberg_0",
    "beethoven_moonlight_0",
    "mozart_requiem_0"
)

internal fun mockTastePerformances(favoriteIds: Set<String>): List<PerformanceItem> =
    mockPerformances.filter { it.id in favoriteIds }

internal fun mockPerformancesFor(work: Work): List<PerformanceItem> =
    mockPerformances.filter { it.workId == work.id }

internal fun mockPerformanceById(id: String): PerformanceItem? =
    mockPerformances.find { it.id == id }
