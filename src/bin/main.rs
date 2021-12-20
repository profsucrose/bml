use std::{env, fs, path::Path, process};

use bml::{preprocessor::PreProcessor, scanner::Scanner, parser::Parser, ast::ast::{eval, Env, Val}};
use lasso::Rodeo;

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
    
    // parse
    let mut parser = Parser::from(&tokens);
    let ast = parser.parse();

    println!("AST: {:#?}", ast);

    let mut env = Env::default();

    env.set(parser.rodeo.get_or_intern("resolution"), Val::Vec2(100.0, 100.0));
    env.set(parser.rodeo.get_or_intern("coord"), Val::Vec2(90.0, 90.0));
    env.set(parser.rodeo.get_or_intern("frag"), Val::Vec4(1.0, 1.0, 1.0, 1.0));

    let result = eval(&ast, env).env.ret;

    println!("Returned: {:?}", result);
    // println!(
    //     "Program: {}",
    //     tokens
    //         .into_iter()
    //         .map(|t| t.lexeme.clone())
    //         .collect::<String>()
    // );
}
