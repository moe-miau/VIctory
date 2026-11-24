use std::path::Path;

pub struct ValidationResult {
    pub exe_found: bool,
    pub exe_name: Option<String>,
    pub rpf_files: Vec<(String, u64)>,
    pub dlls: Vec<String>,
    pub manifest_found: bool,
    pub total_size_gb: f64,
    pub missing: Vec<&'static str>,
}

pub const REQUIRED_RPFS: &[&str] = &[
    "anim.rpf",
    "audio.rpf",
    "common.rpf",
    "data.rpf",
    "levels.rpf",
    "movies.rpf",
    "navmeshes.rpf",
    "packs.rpf",
    "script.rpf",
    "shaders.rpf",
    "textures.rpf",
];

pub fn validate_game_dir(dir: &Path) -> ValidationResult {
    let mut result = ValidationResult {
        exe_found: false,
        exe_name: None,
        rpf_files: Vec::new(),
        dlls: Vec::new(),
        manifest_found: false,
        total_size_gb: 0.0,
        missing: Vec::new(),
    };

    if !dir.is_dir() {
        result.missing.push("directory does not exist");
        return result;
    }

    let entries: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .collect();

    let mut total_bytes: u64 = 0;

    for entry in &entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let meta = entry.metadata().ok();
        let size = meta.as_ref().map_or(0, |m| m.len());
        total_bytes += size;

        if name.ends_with(".exe") && name.contains("scarlett") {
            result.exe_found = true;
            result.exe_name = Some(name.clone());
        }

        if name.ends_with(".rpf") {
            result.rpf_files.push((name.clone(), size));
        }

        if name.ends_with(".dll") {
            result.dlls.push(name.clone());
        }

        if name == "MicrosoftGame.Config" || name == "MicrosoftGame.config" {
            result.manifest_found = true;
        }
    }

    result.total_size_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    if !result.exe_found {
        result.missing.push("game executable (*scarlett*.exe)");
    }
    if !result.manifest_found {
        result.missing.push("MicrosoftGame.Config");
    }

    for required in REQUIRED_RPFS {
        if !result.rpf_files.iter().any(|(n, _)| n == required) {
            result.missing.push(required);
        }
    }

    result
}
