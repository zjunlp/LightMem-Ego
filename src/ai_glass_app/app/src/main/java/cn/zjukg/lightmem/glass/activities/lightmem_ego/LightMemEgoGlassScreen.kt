package cn.zjukg.lightmem.glass.activities.lightmem_ego

import android.Manifest
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LocalLifecycleOwner
import cn.zjukg.lightmem.glass.camera.rememberCameraBound
import cn.zjukg.lightmem.glass.input.BareKeyEvent
import cn.zjukg.lightmem.glass.input.RegisterBareKeyHandler
import cn.zjukg.lightmem.glass.ui.design.BareInfoBlock
import cn.zjukg.lightmem.glass.ui.design.BareKeyGuide
import cn.zjukg.lightmem.glass.ui.design.BareRichInfoBlock
import cn.zjukg.lightmem.glass.ui.design.BareScreenLayout
import cn.zjukg.lightmem.glass.ui.design.BareTokens
import cn.zjukg.lightmem.glass.lightmem_ego.ImageProxyJpegConverter
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoConfig
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoDiagnostics
import cn.zjukg.lightmem.glass.lightmem_ego.LightMemEgoSessionStartMode
import cn.zjukg.lightmem.glass.ui.theme.NeonGreen
import java.util.concurrent.Executors

private const val ANSWER_LINES_PER_PAGE = 6

@Composable
fun LightMemEgoGlassScreen(
    onBack: () -> Unit,
    viewModel: LightMemEgoGlassViewModel,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val state by viewModel.uiState.collectAsState()
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    var hasEnteredSessionScreen by remember { mutableStateOf(false) }
    val shouldBindCamera = state.cameraGranted && state.running
    var answerPageIndex by remember { mutableIntStateOf(0) }
    val answerPages = remember(state.answer) {
        state.answer.toMarkdownAnswerPages(linesPerPage = ANSWER_LINES_PER_PAGE)
    }
    val visibleAnswerPageIndex = if (answerPages.isEmpty()) {
        0
    } else {
        answerPageIndex.coerceIn(0, answerPages.lastIndex)
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grants ->
        viewModel.onPermissionsResult(grants)
    }
    DisposableEffect(Unit) {
        LightMemEgoDiagnostics.log(context, "screen-enter", "LightMemEgoGlassScreen")
        viewModel.refreshPermissions()
        onDispose {
            LightMemEgoDiagnostics.log(context, "screen-dispose", "LightMemEgoGlassScreen running=${state.running}")
            analysisExecutor.shutdown()
            viewModel.stopStreaming()
        }
    }

    LaunchedEffect(state.answer) {
        answerPageIndex = 0
    }

    LaunchedEffect(state.cameraGranted, state.audioGranted) {
        if (!state.cameraGranted || !state.audioGranted) {
            permissionLauncher.launch(
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.RECORD_AUDIO,
                ),
            )
        }
    }

    rememberCameraBound(
        context = context,
        lifecycleOwner = lifecycleOwner,
        enabled = shouldBindCamera,
        onReady = { viewModel.setCameraReady(true) },
        onError = { viewModel.onFrameConvertFailed(it) },
        onUnbind = {
            viewModel.setCameraReady(false)
        },
        useCases = {
            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(LightMemEgoConfig.FRAME_CAPTURE_WIDTH, LightMemEgoConfig.FRAME_CAPTURE_HEIGHT))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analyzer.setAnalyzer(analysisExecutor) { image ->
                handleAnalysisFrame(image, viewModel)
            }
            arrayOf(analyzer)
        },
    )

    RegisterBareKeyHandler { event ->
        when (event) {
            BareKeyEvent.Click -> {
                if (!hasEnteredSessionScreen && !state.running) {
                    viewModel.toggleSessionStartMode()
                } else {
                    viewModel.askSelectedPresetQuestion()
                }
                true
            }
            BareKeyEvent.SpriteClick -> {
                viewModel.toggleVoiceQuestionRecording()
                true
            }
            BareKeyEvent.DoubleClick -> {
                viewModel.selectNextPresetQuestion()
                true
            }
            BareKeyEvent.LongPress -> {
                if (state.running) {
                    hasEnteredSessionScreen = false
                    viewModel.stopStreaming()
                } else {
                    hasEnteredSessionScreen = true
                    viewModel.startStreaming()
                }
                true
            }
            BareKeyEvent.TwoFingerClick -> {
                true
            }
            BareKeyEvent.TwoFingerDoubleClick -> {
                true
            }
            BareKeyEvent.TwoFingerLongPress -> {
                if (answerPages.isNotEmpty()) {
                    answerPageIndex = (visibleAnswerPageIndex + 1) % answerPages.size
                }
                true
            }
            BareKeyEvent.SwipeForward,
            BareKeyEvent.SwipeBack -> false
        }
    }

    val audioQuestionLines = when {
        state.voiceQuestionRecording -> listOf("Listening...")
        state.voiceQuestionStatus == "transcribing" -> listOf("Transcribing...")
        state.voiceQuestionStatus == "asking" && state.voiceQuestionText.isNotBlank() ->
            listOf("Question: ${state.voiceQuestionText}")
        state.voiceQuestionText.isNotBlank() ->
            listOf("Question: ${state.voiceQuestionText}")
        state.voiceQuestionMessage.isNotBlank() -> listOf(state.voiceQuestionMessage)
        state.lastQuestion.isNotBlank() -> listOf("Question: ${state.lastQuestion}")
        else -> listOf("Ready")
    }
    val questionsPerPage = 2
    val pageIndex = state.selectedQuestionIndex / questionsPerPage
    val pageCount = ((state.quickQuestions.size + questionsPerPage - 1) / questionsPerPage).coerceAtLeast(1)
    val firstQuestionIndex = pageIndex * questionsPerPage
    val questionLines = state.quickQuestions
        .drop(firstQuestionIndex)
        .take(questionsPerPage)
        .mapIndexed { offset, question ->
            val questionIndex = firstQuestionIndex + offset
            val marker = if (questionIndex == state.selectedQuestionIndex) ">" else " "
            "$marker ${questionIndex + 1}. ${question.compactQuestion()}"
        }
        .ifEmpty { listOf("No preset questions") }
    val showingAnswer = state.running && answerPages.isNotEmpty()
    val answerLabel = answerLabelFor(
        showingAnswer = showingAnswer,
        currentIndex = visibleAnswerPageIndex,
        pageCount = answerPages.size,
    )
    val answerLine = when {
        state.lastError.isNotBlank() -> "Error: ${state.lastError}"
        !state.running -> "Hold starts capture"
        showingAnswer -> ""
        state.asking -> "Thinking... ${state.queryStatus}"
        !state.memoryReady && !state.canAsk -> "Memory not ready"
        state.lastQuestion.isNotBlank() -> "Asked: ${state.lastQuestion.compactQuestion()}"
        state.canAsk -> "Audio question ready"
        else -> "Question service is not ready"
    }
    val answerLines = if (showingAnswer) {
        answerPages[visibleAnswerPageIndex].map { it.toAnnotatedString() }
    } else {
        answerLine
            .toMarkdownAnswerPages(linesPerPage = ANSWER_LINES_PER_PAGE)
            .firstOrNull()
            ?.map { it.toAnnotatedString() }
            ?: listOf(AnnotatedString(answerLine))
    }
    val latencyLine = state.answerLatencyMs?.let { "Latency: ${formatAnswerLatency(it)}" }.orEmpty()

    if (!hasEnteredSessionScreen && !state.running) {
        BareScreenLayout(
            title = "LightMem-Ego",
            subtitle = null,
            keyGuide = BareKeyGuide(
                click = if (state.hasLastSession) "Switch" else null,
                longPress = "Start",
            ),
            drawSafeAreaFrame = false,
        ) {
            LightMemEgoWelcomeContent(
                state = state,
                onSelectStartMode = viewModel::setSessionStartMode,
            )
        }
        return
    }

    BareScreenLayout(
        title = "LightMem-Ego",
        subtitle = state.displayDayLabel(),
        keyGuide = BareKeyGuide(
            click = "Ask question",
            spriteClick = when {
                state.voiceQuestionRecording -> "Stop voice question"
                state.running -> "Start voice question"
                else -> null
            },
            doubleClick = "Next question",
            twoFingerLongPress = if (answerPages.size > 1) "Next answer" else null,
            longPress = if (state.running || state.sessionId.isNotBlank()) "Stop" else "Start",
        ),
        drawSafeAreaFrame = false,
    ) {
        BareInfoBlock(
            label = "Audio Question",
            lines = audioQuestionLines,
            maxLineCount = 2,
            maxLinesPerItem = 2,
        )
        BareInfoBlock(
            label = "Preset Questions (${pageIndex + 1}/$pageCount)",
            lines = questionLines,
            maxLineCount = questionsPerPage,
        )
        BareRichInfoBlock(
            label = answerLabel,
            lines = answerLines,
            trailingLabel = latencyLine,
            maxLineCount = ANSWER_LINES_PER_PAGE,
            maxLinesPerItem = 1,
        )
        Spacer(modifier = Modifier.weight(1f))
    }
}

@Composable
private fun LightMemEgoWelcomeContent(
    state: LightMemEgoGlassUiState,
    onSelectStartMode: (LightMemEgoSessionStartMode) -> Unit,
) {
    val continueSelected = state.startMode == LightMemEgoSessionStartMode.ContinueLast && state.hasLastSession
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "Hi Ethan!",
                color = NeonGreen,
                fontSize = BareTokens.HeroSp,
                fontWeight = FontWeight.SemiBold,
                lineHeight = BareTokens.HeroSp * 1.2f,
                textAlign = TextAlign.Center,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .padding(horizontal = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            WelcomeSessionOption(
                label = "new session",
                selected = !continueSelected,
                enabled = true,
                modifier = Modifier.weight(1f),
                onClick = { onSelectStartMode(LightMemEgoSessionStartMode.NewSession) },
            )
            WelcomeSessionOption(
                label = "continue",
                selected = continueSelected,
                enabled = state.hasLastSession,
                modifier = Modifier.weight(1f),
                onClick = { onSelectStartMode(LightMemEgoSessionStartMode.ContinueLast) },
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
    }
}

@Composable
private fun WelcomeSessionOption(
    label: String,
    selected: Boolean,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(8.dp)
    val textAlpha = when {
        selected -> 1f
        enabled -> 0.7f
        else -> 0.34f
    }
    Box(
        modifier = modifier
            .height(52.dp)
            .border(
                width = if (selected) 1.dp else 0.dp,
                color = if (selected) NeonGreen else NeonGreen.copy(alpha = 0f),
                shape = shape,
            )
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 8.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = label,
            color = NeonGreen.copy(alpha = textAlpha),
            fontSize = BareTokens.BodySp,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private fun LightMemEgoGlassUiState.displayDayLabel(): String {
    val label = displayDayLabel.trim().ifBlank { dayLabel.trim() }
    if (label.isBlank() || label.equals("null", ignoreCase = true) || label.equals("none", ignoreCase = true)) return "-"
    return label
}

private fun handleAnalysisFrame(image: ImageProxy, viewModel: LightMemEgoGlassViewModel) {
    try {
        if (!viewModel.reserveFrameCapture()) {
            return
        }
        val frame = ImageProxyJpegConverter.toJpegFrame(image)
        if (frame == null) {
            viewModel.onFrameConvertFailed("Frame encode failed")
            return
        }
        viewModel.onFrameConverted(frame.bytes, frame.width, frame.height)
    } catch (error: Exception) {
        viewModel.onFrameConvertFailed(error.message ?: "Frame processing failed")
    } finally {
        image.close()
    }
}

private fun String.compactQuestion(maxChars: Int = 34): String =
    if (length <= maxChars) this else take(maxChars - 1) + "..."

private fun String.voiceStatusLabel(): String = when (this) {
    "idle" -> "idle"
    "recording" -> "recording"
    "transcribing" -> "transcribing"
    "asking" -> "answering"
    "done" -> "done"
    "failed" -> "failed"
    "waiting" -> "waiting"
    else -> this
}
