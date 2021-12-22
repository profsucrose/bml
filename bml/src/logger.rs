use std::process;

use colored::Colorize;

#[derive(Debug)]
pub enum ErrorType {
    Parse,
    Preprocessor,
    Runtime,
}

pub fn info(message: &str) {
    println!("{}", message);
}

pub fn report(error: ErrorType, line: usize, message: &str) -> ! {
    println!(
        "{}{}{}",
        format!("{:?}Error", error).as_str().bright_red().bold(),
        format!(" [line {}]", line).as_str().yellow(),
        format!(": {}", message).as_str()
    );

    println!("{}", "Aborting...".bold());

    process::exit(1)
}
