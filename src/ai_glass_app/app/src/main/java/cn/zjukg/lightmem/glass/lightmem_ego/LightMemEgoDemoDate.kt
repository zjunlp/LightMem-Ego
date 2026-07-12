package cn.zjukg.lightmem.glass.lightmem_ego

import java.time.Clock
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object LightMemEgoDemoDate {
    private val formatter = DateTimeFormatter.ofPattern("yyyy.M.d")

    fun actualDateLabel(clock: Clock = Clock.systemDefaultZone()): String =
        formatter.format(LocalDate.now(clock))
}
