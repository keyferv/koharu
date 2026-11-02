use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use koharu_config::Config;
use koharu_pipeline::{
    Committer, DetectionModel, Flux2KleinConfig, InpaintingModel, KoharuLayoutRFDetrSeg2XLConfig,
    OcrModel, Operation, Pipeline, PipelineConfig, Progress, Request, RoremMixedConfig, Scope,
    StageOutput, TranslationConfig,
};
use koharu_rasterizer::{RasterOptions, Rasterizer};
use koharu_renderer::Renderer;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft, Session};
use koharu_translator::{GenerationConfig, Language, ModelSelection, Provider, ProvidersConfig};

#[derive(Debug, Parser)]
#[command(version, about = "Run Koharu's complete in-process pipeline")]
struct Arguments {
    #[arg(short, long, value_name = "INPUT")]
    input: PathBuf,

    #[arg(short, long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    #[arg(long, value_enum, default_value = "koharu-layout-rfdetr-seg-2xl")]
    detection: DetectionChoice,

    #[arg(long, value_enum, default_value = "hayai-ocr")]
    ocr: OcrChoice,

    #[arg(long, value_enum, default_value = "lama")]
    inpainting: InpaintingChoice,

    #[arg(long, value_enum, default_value = "openai-compatible")]
    provider: ProviderChoice,

    #[arg(long, default_value = "http://127.0.0.1:8765/v1")]
    base_url: Option<String>,

    #[arg(long, default_value = "es")]
    target_language: Language,

    #[arg(long)]
    translation_instructions: Option<String>,

    #[arg(long, default_value = "gpt-6-luna")]
    llm: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    overwrite: bool,

    #[arg(long)]
    limit: Option<usize>,
}

struct SessionCommitter<'a>(&'a mut Session);

#[async_trait::async_trait]
impl Committer for SessionCommitter<'_> {
    async fn commit(&mut self, output: StageOutput) -> Result<koharu_scene::Snapshot> {
        Ok(self.0.commit(output.patch).await?.snapshot)
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DetectionChoice {
    #[value(name = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrChoice {
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[value(name = "manga-ocr")]
    MangaOcr,
    #[value(name = "baberu-ocr")]
    BaberuOcr,
    #[value(name = "hayai-ocr")]
    HayaiOcr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InpaintingChoice {
    #[value(name = "lama")]
    LaMa,
    #[value(name = "aot-inpainting")]
    AotInpainting,
    #[value(name = "flux2-klein")]
    Flux2Klein,
    #[value(name = "rorem-mixed")]
    RoremMixed,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ProviderChoice {
    #[value(name = "local")]
    Local,
    #[value(name = "openai-compatible")]
    OpenAiCompatible,
    #[value(name = "openai")]
    OpenAi,
}

impl Arguments {
    fn pipeline_config(&self) -> PipelineConfig {
        PipelineConfig {
            detection: match self.detection {
                DetectionChoice::KoharuLayoutRFDetrSeg2XL => {
                    DetectionModel::KoharuLayoutRFDetrSeg2XL(
                        KoharuLayoutRFDetrSeg2XLConfig::default(),
                    )
                }
            },
            ocr: match self.ocr {
                OcrChoice::PaddleOcrVl1_6 => OcrModel::PaddleOcrVl1_6,
                OcrChoice::MangaOcr => OcrModel::MangaOcr,
                OcrChoice::BaberuOcr => OcrModel::BaberuOcr,
                OcrChoice::HayaiOcr => OcrModel::HayaiOcr,
            },
            translation: TranslationConfig {
                model: ModelSelection {
                    provider: match self.provider {
                        ProviderChoice::Local => Provider::Local,
                        ProviderChoice::OpenAiCompatible => Provider::OpenAiCompatible,
                        ProviderChoice::OpenAi => Provider::OpenAi,
                    },
                    model: Some(self.llm.clone()),
                    quantization: None,
                    vision: false,
                    reasoning: false,
                },
                generation: GenerationConfig::default(),
                target_language: self.target_language,
                instructions: self.translation_instructions.clone(),
            },
            inpainting: match self.inpainting {
                InpaintingChoice::LaMa => InpaintingModel::LaMa {},
                InpaintingChoice::AotInpainting => InpaintingModel::AotInpainting {},
                InpaintingChoice::Flux2Klein => {
                    InpaintingModel::Flux2Klein(Flux2KleinConfig::default())
                }
                InpaintingChoice::RoremMixed => {
                    InpaintingModel::RoremMixed(RoremMixedConfig::default())
                }
            },
            processor: Default::default(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    initialize_with_retry().await;

    let mut tasks: Vec<(PathBuf, PathBuf)> = Vec::new();
    if arguments.input.is_dir() {
        let out_dir = match &arguments.output {
            Some(out) => out.clone(),
            None => {
                let dir_name = arguments
                    .input
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("output");
                arguments.input.with_file_name(format!("{dir_name}_es"))
            }
        };
        fs::create_dir_all(&out_dir)
            .with_context(|| format!("failed to create output dir {}", out_dir.display()))?;

        let mut files: Vec<PathBuf> = fs::read_dir(&arguments.input)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map_or(false, |ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp"
                        )
                    })
            })
            .collect();
        files.sort();
        if let Some(limit) = arguments.limit {
            files.truncate(limit);
        }

        for file in files {
            let file_name = file.file_name().expect("file has name").to_os_string();
            let target = out_dir.join(file_name);
            tasks.push((file, target));
        }
    } else {
        let target = arguments.output.clone().unwrap_or_else(|| {
            let stem = arguments
                .input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            let ext = arguments
                .input
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("png");
            arguments.input.with_file_name(format!("{stem}_es.{ext}"))
        });
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        tasks.push((arguments.input.clone(), target));
    }

    if tasks.is_empty() {
        eprintln!("No images found to process in {}", arguments.input.display());
        return Ok(());
    }

    let mut providers = ProvidersConfig::default();
    if let Some(base_url) = &arguments.base_url {
        if let Ok(url) = url::Url::parse(base_url) {
            providers.openai_compatible.base_url = Some(url);
        }
    }

    eprintln!(
        "Initializing pipeline (detection={:?}, ocr={:?}, inpainting={:?}, provider={:?}, model={})...",
        arguments.detection, arguments.ocr, arguments.inpainting, arguments.provider, arguments.llm
    );
    let device = koharu_ml::device(arguments.cpu);
    let pipeline = Pipeline::from_config(
        Config::memory(arguments.pipeline_config()),
        Config::memory(providers),
        device,
    )?;
    let renderer = Renderer::new()?;
    let rasterizer = Rasterizer::new()?;

    let total = tasks.len();
    eprintln!("Starting queue: {} images to process", total);
    let batch_start = Instant::now();

    for (idx, (input_path, output_path)) in tasks.iter().enumerate() {
        let current = idx + 1;
        let file_name = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image");

        if output_path.exists() && !arguments.overwrite {
            eprintln!("[{current}/{total}] Skipping {file_name} (already exists)");
            continue;
        }

        eprintln!("[{current}/{total}] Processing {file_name}...");
        let page_start = Instant::now();

        let source = match fs::read(input_path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("[{current}/{total}] Error reading {}: {error}", input_path.display());
                continue;
            }
        };
        let decoded = match image::load_from_memory(&source) {
            Ok(img) => img,
            Err(error) => {
                eprintln!("[{current}/{total}] Error decoding {}: {error}", input_path.display());
                continue;
            }
        };

        let mut session = Session::memory().await?;
        let mut page = None;
        let patch = session.snapshot().patch(|edit| {
            let id = edit.add_page(
                PageDraft::new(
                    file_name,
                    f64::from(decoded.width()),
                    f64::from(decoded.height()),
                ),
                At::End,
            )?;
            edit.set_asset(
                id,
                &AssetRole::new("source")?,
                AssetInput::new(
                    Arc::<[u8]>::from(source),
                    image_media_type(input_path),
                    AssetMetadata {
                        width: Some(decoded.width()),
                        height: Some(decoded.height()),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(id);
            Ok(())
        })?;
        session.commit(patch).await?;
        let page = page.expect("page ID is assigned by the edit");

        let snapshot = session.snapshot();
        let mut committer = SessionCommitter(&mut session);
        let result = pipeline
            .execute(
                snapshot,
                Request {
                    operation: Operation::Full,
                    scope: Scope::Pages(vec![page]),
                    progress: Some(Arc::new(|event| {
                        if let Progress::Finished { stage, elapsed, .. } = event {
                            eprintln!("    [{stage}] finished in {:.2}s", elapsed.as_secs_f64());
                        }
                    })),
                    ..Request::default()
                },
                &mut committer,
            )
            .await;

        if let Err(error) = result {
            eprintln!("[{current}/{total}] Pipeline error on {file_name}: {error}");
            continue;
        }

        let snapshot = session.snapshot();
        let frame = match renderer.render(&snapshot, page).await {
            Ok(f) => f,
            Err(error) => {
                eprintln!("[{current}/{total}] Render error on {file_name}: {error}");
                continue;
            }
        };

        let raster_frame = match frame.raster_frame() {
            Ok(rf) => rf,
            Err(error) => {
                eprintln!("[{current}/{total}] RasterFrame error on {file_name}: {error}");
                continue;
            }
        };

        let raster = match rasterizer.rasterize(&raster_frame, RasterOptions::default()) {
            Ok(r) => r,
            Err(error) => {
                eprintln!("[{current}/{total}] Rasterize error on {file_name}: {error}");
                continue;
            }
        };

        let is_jpeg = output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map_or(false, |ext| {
                matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg")
            });

        let save_result = if is_jpeg {
            let rgb = image::DynamicImage::ImageRgba8(raster.image).to_rgb8();
            rgb.save(output_path)
        } else {
            raster.image.save(output_path)
        };

        if let Err(error) = save_result {
            eprintln!("[{current}/{total}] Save error to {}: {error}", output_path.display());
            continue;
        }

        eprintln!(
            "[{current}/{total}] OK: Saved {} in {:.2}s",
            output_path.display(),
            page_start.elapsed().as_secs_f64()
        );
    }

    eprintln!(
        "Batch translation finished! Total time: {:.2}s",
        batch_start.elapsed().as_secs_f64()
    );
    Ok(())
}

async fn initialize_with_retry() {
    let mut delay = Duration::from_secs(1);
    let mut attempt = 0_u64;
    loop {
        attempt += 1;
        match koharu_ml::init().await {
            Ok(()) => return,
            Err(error) => {
                let jitter = Duration::from_millis((attempt.wrapping_mul(137)) % 251);
                let wait = delay + jitter;
                eprintln!(
                    "runtime initialization attempt {attempt} failed: {error}; retrying in {:.1}s",
                    wait.as_secs_f64()
                );
                tokio::time::sleep(wait).await;
                delay = delay.saturating_mul(2).min(Duration::from_secs(30));
            }
        }
    }
}

fn image_media_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_flags_select_models() {
        let arguments = Arguments::try_parse_from([
            "run",
            "--input",
            "input.png",
            "--output",
            "output.png",
            "--ocr",
            "manga-ocr",
            "--inpainting",
            "flux2-klein",
        ])
        .unwrap();
        assert!(matches!(
            arguments.pipeline_config().ocr,
            OcrModel::MangaOcr
        ));
        assert!(matches!(
            arguments.pipeline_config().inpainting,
            InpaintingModel::Flux2Klein(_)
        ));
    }
}
