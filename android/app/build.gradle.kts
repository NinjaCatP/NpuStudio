plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

android {
    namespace = "com.ninjacatp.npustudio"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.ninjacatp.npustudio.companion"
        minSdk = 26; targetSdk = 34
        versionCode = 4; versionName = "2.2"
    }
    signingConfigs {
        // Sideload signing: a local keystore (see android/README.md). Not for the Play Store.
        create("side") {
            storeFile = file(System.getenv("NPU_KEYSTORE") ?: "${System.getProperty("user.home")}/.android/debug.keystore")
            storePassword = System.getenv("NPU_KEYSTORE_PASS") ?: "android"
            keyAlias = System.getenv("NPU_KEY_ALIAS") ?: "androiddebugkey"
            keyPassword = System.getenv("NPU_KEY_PASS") ?: "android"
        }
    }
    buildTypes {
        release { isMinifyEnabled = false; signingConfig = signingConfigs.getByName("side") }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.1")
    implementation("com.journeyapps:zxing-android-embedded:4.3.0")
    testImplementation("junit:junit:4.13.2")
}
