package cn.zjukg.lightmem.glass.activities.lightmem_ego

internal fun isQuestionReadyAfterUpload(
    backendCanAsk: Boolean,
    frameUploadedCount: Int,
    audioUploadedCount: Int,
): Boolean =
    backendCanAsk || frameUploadedCount > 0 || audioUploadedCount > 0
