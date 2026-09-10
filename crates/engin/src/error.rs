//! 引擎错误类型。

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnginError {
    Uci(String),
    Neural(String),
    Xiangqi(String),
    Internal(&'static str),
}

impl std::fmt::Display for EnginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uci(message) => write!(f, "{message}"),
            Self::Neural(message) => write!(f, "neural: {message}"),
            Self::Xiangqi(message) => write!(f, "{message}"),
            Self::Internal(name) => write!(f, "internal: {name}"),
        }
    }
}

impl std::error::Error for EnginError {}

impl From<String> for EnginError {
    fn from(message: String) -> Self {
        Self::Xiangqi(message)
    }
}
