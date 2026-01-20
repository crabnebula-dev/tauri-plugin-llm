use std::env;
use std::path::PathBuf;
use std::process::Command;

const COMMANDS: &[&str] = &["ping", "stream"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .android_path("android")
        .ios_path("ios")
        .build();

    // Get the project root directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_root = PathBuf::from(&manifest_dir);

    let go_dir = project_root.join("ffi/go");
    let lib_dir = project_root.join("ffi/libs");
    let lib_output = lib_dir.join("libgotproc.a");

    println!("cargo:rerun-if-changed=ffi/go/lib.go");

    std::fs::create_dir_all(&lib_dir).expect("Failed to create lib directory");

    println!("Building Go library with Clang...");
    let status = Command::new("go")
        .args(["build", "-buildmode=c-archive", "-o"])
        .arg(&lib_output)
        .arg("lib.go")
        .current_dir(&go_dir)
        .env("CGO_ENABLED", "1")
        .env("CC", "clang")
        .env("CXX", "clang++")
        .status()
        .expect("Failed to build Go library");

    if !status.success() {
        panic!("Go build failed");
    }

    // Use absolute path for linking
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=gotproc");

    // Link system dependencies
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=m");
    }

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Security");
    }
}
