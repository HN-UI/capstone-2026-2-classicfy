package com.classicfy.app.ui.screen.search

internal data class Work(
    val id: String,
    val composer: String,
    val title: String
)

internal val mockWorks = listOf(
    Work("bach_goldberg", "바흐", "골드베르크 변주곡"),
    Work("bach_cello_suites", "바흐", "무반주 첼로 모음곡"),
    Work("beethoven_moonlight", "베토벤", "피아노 소나타 14번 월광"),
    Work("beethoven_symphony_5", "베토벤", "교향곡 5번 운명"),
    Work("chopin_nocturne", "쇼팽", "녹턴 2번"),
    Work("mozart_requiem", "모차르트", "레퀴엠"),
    Work("debussy_clair_de_lune", "드뷔시", "달빛"),
    Work("vivaldi_four_seasons", "비발디", "사계"),
    Work("rachmaninoff_piano_concerto_2", "라흐마니노프", "피아노 협주곡 2번"),
    Work("tchaikovsky_swan_lake", "차이콥스키", "백조의 호수")
)
