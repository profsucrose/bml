use std::{path::Path, env, process, fs};

use bml::scanner::Scanner;

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
    let mut scanner = Scanner::new(file);
    let tokens = scanner.scan();

    println!("Tokens: {:?}", tokens);
}