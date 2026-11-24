use std::process::Command;

#[test]
fn translate_rejects_zero_workers() {
    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .args(["translate", env!("CARGO_BIN_EXE_victory"), "--jobs", "0"])
        .output()
        .expect("run victory translate");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--jobs"), "{stderr}");
}
