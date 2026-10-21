use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use gliner::api::server::{HttpServer, ServerError};
use gliner::core::settings::{Settings, SettingsError};
use gliner::logger::{Logger, LoggerError};

#[derive(Debug, thiserror::Error)]
enum MainError {
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error(transparent)]
    Logger(#[from] LoggerError),
    #[error(transparent)]
    Server(#[from] ServerError),
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Serve the HTTP API.
    Serve,
}

/// GLiNER bi-encoder inference service: zero-shot named entities over HTTP.
#[derive(Debug, Parser)]
#[command(version)]
struct Cli {
    /// Settings file; without it, `gliner.toml` is read if present.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: CliCommand,
}

impl Cli {
    async fn run(self) -> Result<(), MainError> {
        let settings = Settings::load(self.config.as_deref())?;
        let _logger = Logger::init(&settings.server.log)?;
        match self.command {
            CliCommand::Serve => HttpServer::serve(&settings).await?,
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match Cli::parse().run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gliner: {error}");
            ExitCode::FAILURE
        }
    }
}
