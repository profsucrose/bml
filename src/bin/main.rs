use std::{path::Path, env, process, fs};

use bml::{scanner::Scanner, preprocessor::PreProcessor};

fn main() {
    let mut args = env::args().skip(1);

    if let Some(path) = args.nth(0) {
        run(&Path::new(path.as_str()));
    } else {
        println!("Usage: bml [script]");
        process::exit(1);
    }
}

fn run(path: &Path) {
    let file = fs::read_to_string(path).unwrap(); // TODO: handle unwrap

    let mut scanner = Scanner::from(file);
    let tokens = scanner.scan();

    // preprocess, expand macros
    let mut preprocessor = PreProcessor::from(tokens);
    let tokens = preprocessor.process();

    // println!("Tokens: {:#?}", tokens);
    println!("Program: {}", tokens.into_iter().map(|t| t.lexeme.clone()).collect::<String>());
}