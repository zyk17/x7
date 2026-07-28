//! Engine-wide UCI options, independent from a concrete search implementation.
//!
//! Mirrors px0 `OptionsDict` ownership in `src/engine.h`: `Engine` keeps the
//! options and passes the relevant snapshot to its factory-produced search.

/// Minimal formal UCI option set for the stream migration.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Options {
    pub weights_file: String,
    pub show_wdl: bool,
    pub show_eps: bool,
}

impl Options {
    pub fn populate_defaults() -> Self {
        Self::default()
    }

    pub fn list_options_uci(&self) -> Vec<String> {
        vec![
            format!("option name UCI_ShowWDL type check default {}", self.show_wdl),
            format!("option name UCI_ShowEPS type check default {}", self.show_eps),
            format!("option name WeightsFile type string default {}", self.weights_file),
        ]
    }

    pub fn set_uci_option(&mut self, name: &str, value: &str) -> Result<(), crate::EnginError> {
        let flag = |name| match value {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(crate::EnginError::Uci(format!(
                "Flag '{name}' must be either true or false"
            ))),
        };
        match name {
            "UCI_ShowWDL" => self.show_wdl = flag(name)?,
            "UCI_ShowEPS" => self.show_eps = flag(name)?,
            "WeightsFile" => self.weights_file = value.to_owned(),
            _ => return Err(crate::EnginError::Uci(format!("Unknown option: {name}"))),
        }
        Ok(())
    }
}
