//! px0 `utils/exception.h` 在 UCI 层的 Rust 映射。

use xiangqi_core::CoreError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnginError {
    Uci(String),
    Core(CoreError),
    PortIncomplete(&'static str),
}

impl std::fmt::Display for EnginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uci(message) => write!(f, "{message}"),
            Self::Core(error) => write!(f, "{error}"),
            Self::PortIncomplete(name) => write!(f, "px0 port incomplete: {name}"),
        }
    }
}

impl std::error::Error for EnginError {}

impl From<CoreError> for EnginError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}
