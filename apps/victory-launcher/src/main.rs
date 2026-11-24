#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod config;
mod validate;

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use eframe::egui;

use config::{Console, LauncherConfig};
use validate::ValidationResult;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 580.0])
            .with_min_inner_size([600.0, 480.0])
            .with_title("VIctory Launcher"),
        ..Default::default()
    };
    eframe::run_native(
        "VIctory Launcher",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}

#[derive(PartialEq, Eq)]
enum Tab {
    Launch,
    Settings,
    Log,
}

struct App {
    cfg: LauncherConfig,
    validation: Option<ValidationResult>,
    tab: Tab,
    game_dir_text: String,
    log_output: Arc<Mutex<String>>,
    child: Option<Child>,
    running: bool,
    status_msg: String,
    victory_exe: Option<PathBuf>,
}

impl App {
    fn new() -> Self {
        let cfg = LauncherConfig::load();
        let game_dir_text = cfg
            .game_dir
            .as_ref()
            .map_or_else(String::new, |p| p.display().to_string());
        let validation = cfg
            .game_dir
            .as_ref()
            .map(|p| validate::validate_game_dir(p));
        let mut app = Self {
            cfg,
            validation,
            tab: Tab::Launch,
            game_dir_text,
            log_output: Arc::new(Mutex::new(String::new())),
            child: None,
            running: false,
            status_msg: String::new(),
            victory_exe: None,
        };
        app.refresh_exe();
        app
    }

    fn refresh_exe(&mut self) {
        self.victory_exe = find_victory_exe(self.cfg.diagnostics);
        self.status_msg = match (&self.victory_exe, self.cfg.diagnostics) {
            (Some(_), _) => String::new(),
            (None, false) => {
                "victory.exe not found - build with: cargo build --release -p victory".into()
            }
            (None, true) => "diagnostics victory.exe not found - build with: \
                 cargo build --profile diag -p victory --features diagnostics"
                .into(),
        };
    }

    fn revalidate(&mut self) {
        self.cfg.game_dir = if self.game_dir_text.is_empty() {
            None
        } else {
            Some(PathBuf::from(&self.game_dir_text))
        };
        self.validation = self
            .cfg
            .game_dir
            .as_ref()
            .map(|p| validate::validate_game_dir(p));
    }

    fn launch(&mut self) {
        let Some(ref victory) = self.victory_exe else {
            self.status_msg = "victory.exe not found".into();
            return;
        };
        let Some(ref game_dir) = self.cfg.game_dir else {
            self.status_msg = "No game directory set".into();
            return;
        };

        let validation = validate::validate_game_dir(game_dir);
        if !validation.missing.is_empty() {
            self.status_msg = format!("Missing files: {}", validation.missing.join(", "));
            return;
        }

        let exe_name = validation
            .exe_name
            .unwrap_or_else(|| "game_scarlett_master.exe".into());
        let game_exe = game_dir.join(&exe_name);
        let game_exe = std::path::absolute(&game_exe).unwrap_or(game_exe);

        let config_path = match self.cfg.write_victory_toml() {
            Ok(p) => p,
            Err(e) => {
                self.status_msg = format!("Failed to write config: {e}");
                return;
            }
        };

        let mut cmd = Command::new(victory);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        cmd.arg("run")
            .arg(&game_exe)
            .arg("--execute")
            .arg("--console")
            .arg(match self.cfg.console {
                Console::SeriesX => "series-x",
                Console::SeriesS => "series-s",
            })
            .env("VICTORY_CONFIG", &config_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if self.cfg.use_warp {
            cmd.env("VICTORY_D3D12_WARP", "1");
        }
        // Both are read only by a diagnostics build.
        if self.cfg.diagnostics {
            if self.cfg.d3d12_debug {
                cmd.env("VICTORY_D3D12_DEBUG", "1");
            }
            if let Some(ref skip) = self.cfg.skip_wave64 {
                cmd.env("VICTORY_SKIP_WAVE64", skip);
            }
        }

        {
            let mut log = self.log_output.lock().unwrap();
            log.clear();
            log.push_str(&format!(
                "--- VIctory Launcher ---\n\
                 Executable: {}\n\
                 Game: {}\n\
                 Console: {}\n\
                 WARP: {}\n\
                 Diagnostics build: {}\n\
                 ---\n\n",
                victory.display(),
                game_exe.display(),
                self.cfg.console,
                self.cfg.use_warp,
                self.cfg.diagnostics,
            ));
        }

        match cmd.spawn() {
            Ok(mut child) => {
                if let Some(stdout) = child.stdout.take() {
                    spawn_reader(stdout, Arc::clone(&self.log_output));
                }
                if let Some(stderr) = child.stderr.take() {
                    spawn_reader(stderr, Arc::clone(&self.log_output));
                }
                self.child = Some(child);
                self.running = true;
                self.status_msg = "Game launched!".into();
                self.tab = Tab::Log;
            }
            Err(e) => {
                self.status_msg = format!("Failed to launch: {e}");
            }
        }
    }

    fn check_child(&mut self) {
        if let Some(ref mut child) = self.child {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.running = false;
                    self.child = None;
                    let log_path = LauncherConfig::config_path().with_file_name("last-run.log");
                    let saved = std::fs::write(&log_path, &*self.log_output.lock().unwrap());
                    self.status_msg = match saved {
                        Ok(()) => format!(
                            "Game exited with {status} - log saved to {}",
                            log_path.display()
                        ),
                        Err(_) => format!("Game exited with {status}"),
                    };
                }
                Ok(None) => {}
                Err(e) => {
                    self.running = false;
                    self.status_msg = format!("Error checking game: {e}");
                    self.child = None;
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_child();
        if self.running {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VIctory");
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Launch, "Launch");
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.selectable_value(&mut self.tab, Tab::Log, "Log");
            });
        });

        if !self.status_msg.is_empty() {
            egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                let color = if self.status_msg.contains("Missing")
                    || self.status_msg.contains("Failed")
                    || self.status_msg.contains("not found")
                {
                    egui::Color32::from_rgb(220, 60, 60)
                } else if self.running {
                    egui::Color32::from_rgb(60, 180, 60)
                } else {
                    ui.visuals().text_color()
                };
                ui.colored_label(color, &self.status_msg);
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Launch => self.draw_launch(ui),
            Tab::Settings => self.draw_settings(ui),
            Tab::Log => self.draw_log(ui),
        });
    }
}

impl App {
    fn draw_launch(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);

        // Game directory.
        ui.heading("Game files");
        ui.horizontal(|ui| {
            ui.label("Game directory:");
            let changed = ui.text_edit_singleline(&mut self.game_dir_text).changed();
            if ui.button("Browse...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .set_title("Select game directory")
                    .pick_folder()
            {
                self.game_dir_text = path.display().to_string();
                self.revalidate();
                let _ = self.cfg.save();
            }
            if changed {
                self.revalidate();
            }
        });

        ui.add_space(8.0);

        // Validation status.
        if let Some(ref v) = self.validation {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.heading("Validation");
                egui::Grid::new("validation_grid")
                    .num_columns(2)
                    .spacing([16.0, 4.0])
                    .show(ui, |ui| {
                        let check = |ok: bool| {
                            if ok { "\u{2714}" } else { "\u{2716}" }
                        };

                        ui.label("Executable:");
                        ui.label(format!(
                            "{} {}",
                            check(v.exe_found),
                            v.exe_name.as_deref().unwrap_or("not found")
                        ));
                        ui.end_row();

                        ui.label("Manifest:");
                        ui.label(format!("{} MicrosoftGame.Config", check(v.manifest_found)));
                        ui.end_row();

                        ui.label("RPF archives:");
                        let required_found = validate::REQUIRED_RPFS
                            .iter()
                            .filter(|r| v.rpf_files.iter().any(|(n, _)| n == *r))
                            .count();
                        ui.label(format!(
                            "{} {} / {} required ({} total)",
                            check(required_found == validate::REQUIRED_RPFS.len()),
                            required_found,
                            validate::REQUIRED_RPFS.len(),
                            v.rpf_files.len()
                        ));
                        ui.end_row();

                        ui.label("DLLs:");
                        ui.label(format!("{} found", v.dlls.len()));
                        ui.end_row();

                        ui.label("Total size:");
                        ui.label(format!("{:.1} GB", v.total_size_gb));
                        ui.end_row();
                    });

                if !v.missing.is_empty() {
                    ui.add_space(4.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 60, 60),
                        format!("Missing: {}", v.missing.join(", ")),
                    );
                }
            });
        }

        ui.add_space(16.0);

        // Console selector.
        ui.horizontal(|ui| {
            ui.label("Console:");
            ui.radio_value(&mut self.cfg.console, Console::SeriesX, "Series X");
            ui.radio_value(&mut self.cfg.console, Console::SeriesS, "Series S");
        });

        ui.add_space(16.0);

        // Launch button.
        let can_launch = self.victory_exe.is_some()
            && self.cfg.game_dir.is_some()
            && self
                .validation
                .as_ref()
                .is_some_and(|v| v.missing.is_empty())
            && !self.running;

        let button = egui::Button::new(if self.running {
            "Running..."
        } else {
            "Launch GTA VI"
        })
        .min_size(egui::vec2(200.0, 48.0));

        ui.vertical_centered(|ui| {
            if ui.add_enabled(can_launch, button).clicked() {
                let _ = self.cfg.save();
                self.launch();
            }
        });

        if self.running {
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                ui.spinner();
                if ui.button("Stop game").clicked()
                    && let Some(child) = self.child.as_mut()
                    && let Err(e) = child.kill()
                {
                    self.status_msg = format!("Failed to stop game: {e}");
                }
            });
        }
    }

    fn draw_settings(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("Directories");

        ui.horizontal(|ui| {
            ui.label("Cache dir:");
            let mut text = self
                .cfg
                .cache_dir
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string());
            if ui.text_edit_singleline(&mut text).changed() {
                self.cfg.cache_dir = if text.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(&text))
                };
            }
            if ui.button("Browse...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .set_title("Select cache directory")
                    .pick_folder()
            {
                self.cfg.cache_dir = Some(path);
            }
        });

        ui.horizontal(|ui| {
            ui.label("Data dir:");
            let mut text = self
                .cfg
                .data_dir
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string());
            if ui.text_edit_singleline(&mut text).changed() {
                self.cfg.data_dir = if text.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(&text))
                };
            }
            if ui.button("Browse...").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .set_title("Select data directory")
                    .pick_folder()
            {
                self.cfg.data_dir = Some(path);
            }
        });

        ui.add_space(16.0);
        ui.heading("GPU");

        ui.checkbox(
            &mut self.cfg.use_warp,
            "Use WARP software adapter (safe, slow rendering)",
        );
        ui.add_space(16.0);
        ui.heading("Diagnostics");
        if ui
            .checkbox(
                &mut self.cfg.diagnostics,
                "Use the diagnostics build (target\\diag\\victory.exe)",
            )
            .changed()
        {
            self.refresh_exe();
        }

        ui.add_enabled_ui(self.cfg.diagnostics, |ui| {
            ui.checkbox(
                &mut self.cfg.d3d12_debug,
                "Enable D3D12 debug layer (verbose GPU errors)",
            );
            let mut skip_text = self.cfg.skip_wave64.clone().unwrap_or_default();
            ui.horizontal(|ui| {
                ui.label("Skip wave64 stages:");
                if ui.text_edit_singleline(&mut skip_text).changed() {
                    let trimmed = skip_text.trim();
                    self.cfg.skip_wave64 = (!trimmed.is_empty()).then(|| trimmed.to_owned());
                }
            });
            ui.label(
                egui::RichText::new(
                    "Comma-separated: mesh, compute, vertex, pixel (omits wave64 draws)",
                )
                .small()
                .weak(),
            );
        });

        ui.add_space(16.0);
        if ui.button("Save settings").clicked() {
            match self.cfg.save() {
                Ok(()) => self.status_msg = "Settings saved".into(),
                Err(e) => self.status_msg = format!("Save failed: {e}"),
            }
        }

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!(
                "Config: {}",
                LauncherConfig::config_path().display()
            ))
            .small()
            .weak(),
        );
    }

    fn draw_log(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Output");
            if ui.button("Clear").clicked() {
                self.log_output.lock().unwrap().clear();
            }
            if ui.button("Copy").clicked() {
                let text = self.log_output.lock().unwrap().clone();
                ui.ctx().copy_text(text);
            }
        });

        let log = self.log_output.lock().unwrap().clone();
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut log.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_width(f32::INFINITY),
                );
            });
    }
}

const MAX_LOG_BYTES: usize = 4 * 1024 * 1024;

fn spawn_reader(stream: impl std::io::Read + Send + 'static, log: Arc<Mutex<String>>) {
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            let mut l = log.lock().unwrap();
            l.push_str(&line);
            l.push('\n');
            if l.len() > MAX_LOG_BYTES {
                let mut cut = l.len() - MAX_LOG_BYTES / 2;
                while !l.is_char_boundary(cut) {
                    cut += 1;
                }
                l.drain(..cut);
            }
        }
    });
}

/// The diagnostics build lives in `target/diag`, the launcher normally in
/// `target/release`, so "beside the launcher" only fits the release build.
fn find_victory_exe(diagnostics: bool) -> Option<PathBuf> {
    let launcher_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from));
    let mut candidates = Vec::new();
    if diagnostics {
        if let Some(dir) = &launcher_dir {
            candidates.push(dir.join("victory-diag.exe"));
            if let Some(target) = dir.parent() {
                candidates.push(target.join("diag").join("victory.exe"));
            }
        }
        candidates.push(PathBuf::from("target/diag/victory.exe"));
    } else {
        if let Some(dir) = &launcher_dir {
            candidates.push(dir.join("victory.exe"));
        }
        candidates.push(PathBuf::from("target/release/victory.exe"));
        candidates.push(PathBuf::from("target/debug/victory.exe"));
        if let Some(paths) = std::env::var_os("PATH") {
            candidates.extend(std::env::split_paths(&paths).map(|d| d.join("victory.exe")));
        }
    }
    candidates
        .into_iter()
        .find(|c| c.is_file())
        .and_then(|c| std::path::absolute(c).ok())
}
