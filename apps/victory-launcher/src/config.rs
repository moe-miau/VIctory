use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LauncherConfig {
    pub game_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub console: Console,
    pub use_warp: bool,
    pub diagnostics: bool,
    pub d3d12_debug: bool,
    pub skip_wave64: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Console {
    SeriesX,
    SeriesS,
}

impl Default for LauncherConfig {
    fn default() -> Self {
        Self {
            game_dir: None,
            cache_dir: None,
            data_dir: None,
            console: Console::SeriesX,
            use_warp: false,
            diagnostics: false,
            d3d12_debug: false,
            skip_wave64: None,
        }
    }
}

impl LauncherConfig {
    pub fn config_path() -> PathBuf {
        let local = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        local.join("victory").join("launcher.toml")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| toml::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    /// Writes the `victory.toml` the CLI reads (via `VICTORY_CONFIG`) beside
    /// the launcher's own settings, and returns its path.
    pub fn write_victory_toml(&self) -> anyhow::Result<PathBuf> {
        let path = Self::config_path().with_file_name("victory.toml");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, self.victory_toml_text()?)?;
        Ok(path)
    }

    fn victory_toml_text(&self) -> anyhow::Result<String> {
        #[derive(Serialize)]
        struct Paths<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            game_dir: Option<&'a PathBuf>,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_dir: Option<&'a PathBuf>,
            #[serde(skip_serializing_if = "Option::is_none")]
            data_dir: Option<&'a PathBuf>,
        }
        #[derive(Serialize)]
        struct File<'a> {
            paths: Paths<'a>,
        }

        Ok(toml::to_string_pretty(&File {
            paths: Paths {
                game_dir: self.game_dir.as_ref(),
                cache_dir: self.cache_dir.as_ref(),
                data_dir: self.data_dir.as_ref(),
            },
        })?)
    }
}

impl std::fmt::Display for Console {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Console::SeriesX => write!(f, "Series X"),
            Console::SeriesS => write!(f, "Series S"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn victory_toml_keeps_windows_paths() {
        #[derive(Deserialize)]
        struct File {
            paths: Paths,
        }
        #[derive(Deserialize)]
        struct Paths {
            game_dir: PathBuf,
            cache_dir: Option<PathBuf>,
        }

        let dir = PathBuf::from(r"E:\GTA6\GTA6.Xbox.v1.0.2.1");
        let cfg = LauncherConfig {
            game_dir: Some(dir.clone()),
            ..Default::default()
        };
        let text = cfg.victory_toml_text().unwrap();
        let parsed: File = toml::from_str(&text).unwrap();
        assert_eq!(parsed.paths.game_dir, dir);
        assert!(parsed.paths.cache_dir.is_none());
    }
}
