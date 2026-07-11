# LightMem Glass App

This is the Android glasses-side app for Rokid AI Glass. It captures camera frames and microphone audio from the glasses, sends the live stream to a configured LightMem API service, and displays voice-question answers on the glasses screen.

The app uses standard Android APIs, Jetpack Compose UI, CameraX frame capture, `AudioRecord` microphone capture, RootEncoder RTMP streaming, and Rokid touchpad / button input. It does not require a phone-side SDK at runtime.

## Features

- Start and stop a real-time glasses capture session.
- Capture camera frames from the glasses camera.
- Capture microphone audio from the glasses microphone.
- Push live video through RTMP when a `push_url` is available.
- Fall back to HTTP frame/audio upload when RTMP is unavailable.
- Ask short voice questions from the glasses.
- Show answers on a 480 x 640 high-contrast glasses UI.

This open-source version does not include local session recording, replay-from-file mode, preset-question UI, or standalone photo/video/audio/IMU sample screens.

## Project Layout

```text
src/ai_glass_app/
  app/src/main/java/cn/zjukg/lightmem/glass/
    activities/main/       # Android entry activity
    activities/worldmm/    # Glasses UI and session state
    camera/                # CameraX binding helper
    input/                 # Rokid key and touchpad input dispatcher
    ui/design/             # Glasses-oriented UI components
    ui/theme/              # Compose theme
    worldmm/               # API client, RTMP streamer, audio/image helpers
  app/src/main/AndroidManifest.xml
  gradle/libs.versions.toml
```

## Requirements

- Rokid AI Glass running Android 12 API 31 or later.
- Android Studio or Android SDK command-line tools.
- JDK compatible with the Android Gradle Plugin used by this project.
- ADB access to the glasses.

Project settings:

- `minSdk = 31`
- `targetSdk = 36`
- application id: `cn.zjukg.lightmem.glass`

## Configure

Edit:

```text
app/src/main/java/cn/zjukg/lightmem/glass/worldmm/WorldMMConfig.kt
```

Important values:

```kotlin
const val API_BASE_URL = "https://lightmem-ego.zjukg.cn/api"
const val INPUT_MODE = "rokid_live_rtmp"
const val FALLBACK_INPUT_MODE = "rokid_frame_audio"
```

Change `API_BASE_URL` before building if you want the app to connect to a different LightMem API service.

## Build

Run commands from the `src/ai_glass_app/` directory.

Windows:

```powershell
.\gradlew.bat assembleDebug
```

macOS or Linux:

```bash
./gradlew assembleDebug
```

The debug APK is generated at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## Install And Start

1. Enable ADB for the Rokid AI Glass.
2. Check that the device is visible:

```bash
adb devices
```

3. Install the APK:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

4. Start the app from the glasses launcher, or start it with ADB:

```bash
adb shell monkey -p cn.zjukg.lightmem.glass 1
```

5. Watch logs if needed:

```bash
adb logcat | grep OmniSparkDiag
```

On Windows PowerShell:

```powershell
adb logcat | findstr OmniSparkDiag
```

## Controls

- Long press: start or stop the real-time capture session.
- Sprite click while running: start recording a voice question. Click again to stop recording and submit it.
- One-finger click / Enter key: show the next answer page when an answer has multiple pages.
- Two-finger double click: show the previous answer page when an answer has multiple pages.
- Back / one-finger double click: consumed by the app to avoid accidental exit during glasses use.

## Permissions

The app declares only the permissions needed by the glasses-side real-time flow:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

- `CAMERA`: captures frames from the glasses camera.
- `RECORD_AUDIO`: captures microphone audio and voice questions.
- `INTERNET`: sends data to the configured API service.

No external-storage permission is required. Android automatic backup is disabled with `android:allowBackup="false"`.

## Privacy

When a capture session is running, the app captures camera frames and microphone audio and sends them to the configured API service. The current open-source version does not save local session recordings.

## Test

Run unit tests from the `src/ai_glass_app/` directory:

```powershell
.\gradlew.bat testDebugUnitTest
```

Build a debug APK:

```powershell
.\gradlew.bat assembleDebug
```
