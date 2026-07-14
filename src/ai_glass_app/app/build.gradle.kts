import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.isFile) {
        file.inputStream().use(::load)
    }
}

fun releaseProperty(name: String): String? =
    localProperties.getProperty(name)
        ?: providers.gradleProperty(name).orNull
        ?: providers.environmentVariable(name).orNull

android {
    namespace = "cn.zjukg.lightmem.glass"
    buildToolsVersion = "36.1.0"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "cn.zjukg.lightmem.glass"
        minSdk = 31
        targetSdk = 36
        versionCode = 2
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    signingConfigs {
        create("release") {
            val storePath = releaseProperty("LIGHTMEM_RELEASE_STORE_FILE")
            if (!storePath.isNullOrBlank()) {
                storeFile = rootProject.file(storePath)
            }
            storePassword = releaseProperty("LIGHTMEM_RELEASE_STORE_PASSWORD")
            keyAlias = releaseProperty("LIGHTMEM_RELEASE_KEY_ALIAS") ?: "lightmem-ego-release"
            keyPassword = releaseProperty("LIGHTMEM_RELEASE_KEY_PASSWORD")
            enableV1Signing = true
            enableV2Signing = true
            enableV3Signing = true
            enableV4Signing = true
        }
    }

    buildTypes {
        debug {
            buildConfigField("boolean", "LIGHTMEM_DEBUG_ROUTER", "true")
        }
        release {
            isDebuggable = false
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
            buildConfigField("boolean", "LIGHTMEM_DEBUG_ROUTER", "false")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
}

tasks.register("validateReleaseSigning") {
    doLast {
        val required = listOf(
            "LIGHTMEM_RELEASE_STORE_FILE",
            "LIGHTMEM_RELEASE_STORE_PASSWORD",
            "LIGHTMEM_RELEASE_KEY_ALIAS",
            "LIGHTMEM_RELEASE_KEY_PASSWORD",
        )
        val missing = required.filter { releaseProperty(it).isNullOrBlank() }
        if (missing.isNotEmpty()) {
            throw GradleException(
                "Missing release signing properties: ${missing.joinToString()}. " +
                    "Set them in local.properties, gradle.properties, or environment variables."
            )
        }
        val storePath = releaseProperty("LIGHTMEM_RELEASE_STORE_FILE").orEmpty()
        val store = rootProject.file(storePath)
        if (!store.isFile) {
            throw GradleException("Release signing store file does not exist: ${store.absolutePath}")
        }
    }
}

tasks.matching { it.name == "assembleRelease" || it.name == "bundleRelease" }.configureEach {
    dependsOn("validateReleaseSigning")
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    testImplementation(libs.junit)
    testImplementation("org.json:json:20240303")
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
