use std::{env, fs, path::Path, process};

use bml::{preprocessor::PreProcessor, scanner::Scanner, parser::Parser, ast::{eval, Env, Val, Ast}, logger::{info, report, ErrorType}};
use image::{ImageBuffer, DynamicImage, RgbImage, RgbaImage};

fn main() {
    let mut args = env::args().skip(1);

    if let Some(script) = args.next() {
        if let Some(image) = args.next() {
            run(&Path::new(script.as_str()), Some(&Path::new(image.as_str())))
        } else {
            run(&Path::new(script.as_str()), None)
        }

        return;
    }

    info("Usage: bml [image] [script]");
    process::exit(1);
}

fn run(path: &Path, image: Option<&Path>) {
    let file = fs::read_to_string(path).unwrap(); // TODO: handle unwrap

    let mut scanner = Scanner::from(file);
    let tokens = scanner.scan();

    // preprocess, expand macros
    let mut preprocessor = PreProcessor::from(tokens);
    let tokens = preprocessor.process();

    // println!(
    //     "Program: {}",
    //     tokens
    //         .into_iter()
    //         .map(|t| t.lexeme.clone())
    //         .collect::<String>()
    // );

    // parse
    let mut parser = Parser::from(&tokens);
    let ast = parser.parse();

    // println!("AST: {:#?}", ast);

    let mut imgbuf = image
        .map(|x| image::open(x).unwrap().to_rgba8())
        .unwrap_or_else(|| ImageBuffer::new(100, 100));

    process_image(&mut imgbuf, ast, parser);

    imgbuf.save("output.png").unwrap();

    // info(format!("return={:?}", result).as_str());
}

fn process_image(img: &mut RgbaImage, ast: Ast, mut parser: Parser) {
    let (width, height) = img.dimensions();

    let res = parser.rodeo.get_or_intern("resolution");
    let coord = parser.rodeo.get_or_intern("coord");
    let frag = parser.rodeo.get_or_intern("frag");
    let frame = parser.rodeo.get_or_intern("frame");
    let max_frame = parser.rodeo.get_or_intern("max_frame");

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let mut env = Env::default();

        env.set(res, Val::Vec2(width as f32, height as f32));

        // make (0, 0) bottom-left, `image` treats (0, 0) as top-left
        env.set(coord, Val::Vec2((x + 1) as f32, (height - (y + 1)) as f32));

        env.set(frag, Val::Vec4(1.0, 1.0, 1.0, 1.0));

        // TODO: frames
        env.set(frame, Val::Float(0.0));
        env.set(max_frame, Val::Float(0.0));

        match eval(&ast, env).env.ret {
            Some(Val::Vec4(r, g, b, a)) => pixel.0 = [ (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8],
            Some(val) => report(ErrorType::Runtime, ast.line, format!("Expected vec4 from shader, got {:?}", val).as_str()),
            None => report(ErrorType::Runtime, ast.line, "Expected vec4 from shader, got null"),
        }
    }
}