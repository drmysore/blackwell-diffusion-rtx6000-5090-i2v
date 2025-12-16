{ inputs, ... }:
{
  perSystem =
    {
      system,
      pkgs,
      config,
      ...
    }:
    {
      # TODO[b7r6]: this is either too much or too little..
      packages.android-sdk = inputs.android-nixpkgs.sdk.${system} (sdkPkgs: [
        sdkPkgs.build-tools-34-0-0
        sdkPkgs.build-tools-35-0-0
        sdkPkgs.build-tools-36-0-0
        sdkPkgs.cmake-3-22-1
        sdkPkgs.cmdline-tools-latest
        sdkPkgs.emulator
        sdkPkgs.ndk-26-1-10909125
        sdkPkgs.ndk-27-1-12297006
        sdkPkgs.platform-tools
        sdkPkgs.platforms-android-35
        sdkPkgs.platforms-android-36
        sdkPkgs.system-images-android-35-google-apis-x86-64
        sdkPkgs.system-images-android-36-google-apis-x86-64
      ]);

      devShells.android = pkgs.mkShell rec {
        JAVA_HOME = pkgs.corretto17.home;
        ANDROID_HOME = "${config.packages.android-sdk}/share/android-sdk";
        ANDROID_SDK_ROOT = "${config.packages.android-sdk}/share/android-sdk";
        ANDROID_NDK_ROOT = "${config.packages.android-sdk}/share/android-sdk/ndk/27.1.12297006";
        GRADLE_OPTS = "-Dorg.gradle.project.android.aapt2FromMavenOverride=${ANDROID_SDK_ROOT}/build-tools/36.0.0/aapt2";
        ANDROID_AVD_HOME = ''$HOME/.config/.android/avd'';
        ANDROID_USER_HOME = ''$HOME/.android'';

        packages = [
          pkgs.aapt
          pkgs.bun
          pkgs.corretto17
          pkgs.nodejs_24
          pkgs.watchman
          config.packages.android-sdk
        ];

        shellHook = ''
          export GRADLE_USER_HOME="$PWD/.gradle"
          mkdir -p $GRADLE_USER_HOME
          mkdir -p $HOME/.android/cache

          export PATH="${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:$PATH"
          export PATH="${ANDROID_SDK_ROOT}/platform-tools:$PATH"
          export PATH="${ANDROID_SDK_ROOT}/emulator:$PATH"
          export ANDROID_SDK_ROOT_OVERRIDE=$ANDROID_SDK_ROOT

          echo "Android SDK: $ANDROID_SDK_ROOT"
          echo "Android NDK: $ANDROID_NDK_ROOT"
          echo "Gradle Home: $GRADLE_USER_HOME"
        '';
      };
    };
}
