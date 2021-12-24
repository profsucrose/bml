use std::process;

use ast::Val;
use bml::{ast::{self, eval}, logger::{ErrorType, report}};
use image::io::Reader as ImageReader;

macro_rules! gen_runtime_idents {
    ($($x:ident $(,)? )*) => {
        struct RuntimeIdents { $($x: lasso::Spur,)* }
            impl RuntimeIdents {
                fn new(rodeo: &mut lasso::Rodeo<lasso::Spur>) -> Self {
                    Self { $($x: rodeo.get_or_intern(stringify!($x)), )* }
                }
            }
        }
}

gen_runtime_idents!(resolution, coord, frag, frame, frame_count);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let script_name = args.next().expect("Expected script name");
    let img_path = args.next().expect("Expected image path");
    let eval_flag = args.next();

    if args.next().is_some() {
        println!("Usage: bml [image] [script] <options>");
        process::exit(1);
    }

    let (mut rodeo, ast) = {
        let file = std::fs::read_to_string(&script_name)?;
        let raw = bml::Scanner::from(file).scan();
        let expanded = bml::PreProcessor::from(raw).process();
        println!(
            "{}",
            expanded
                .iter()
                .map(|x| format!("{} ", x.lexeme.to_owned()))
                .collect::<String>()
        );
        bml::Parser::from(&expanded).parse()
    };

    let rti = RuntimeIdents::new(&mut rodeo);
    let mut env = ast::Env::default();

    if let Some(flag) = eval_flag {
        if flag == "--eval" {
            let output = eval(&ast, env, &rodeo).env.ret.take();

            println!("return={:?}", output);

            process::exit(0);
        } else {
            println!("Invalid flag '{}' (expected '--eval')", flag);

            process::exit(1);
        }
    }

    let frame_count = std::env::var("BML_FRAME_COUNT")
        .ok()
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(1);

    let mut image = ImageReader::open(&img_path)?.decode()?;

    let buffer = image
        .as_mut_rgba8()
        .expect("Couldn't get pixel buffer from image");

    let (width, height) = buffer.dimensions();

    env.set(rti.resolution, Val::Vec2(width as _, height as _));
    env.set(rti.frame_count, Val::Float(frame_count as _));

    let mut frames = Vec::with_capacity(frame_count);

    (0..frame_count).fold(env, |mut env, frame| {
        env.set(rti.frame, Val::Float(frame as _));

        let mut frame = Vec::<u8>::with_capacity((width * height * 4) as usize);
        let ret = buffer
            .enumerate_pixels()
            .fold(env, |mut env, (x, y, rgba)| {
                let [r, g, b, a] = rgba.0;

                env.set(rti.coord, Val::Vec2(x as _, y as _));
                env.set(
                    rti.frag,
                    Val::Vec4(
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ),
                );

                let mut ret = ast::eval(&ast, env, &rodeo);

                match ret.env.ret.take() {
                    Some(Val::Vec4(x, y, z, w)) => {
                        // println!("{:?}, {:?}", ret.env.get(2), ret.env.get(5));
                        // println!("{:?}, {:?}", ret.env.get(5), ret.env.get(6));
                        // println!("{}, {}, {}, {}", x, y, z, w);
                        frame.push((255.0 * x) as u8);
                        frame.push((255.0 * y) as u8);
                        frame.push((255.0 * z) as u8);
                        frame.push((255.0 * w) as u8);
                    }
                    None => report(ErrorType::Runtime, ast.line, "Expected program to return vec4, got nothing"),
                    Some(val) => report(ErrorType::Runtime, ast.line, format!("Returned color must be vec4, got {:?}", val).as_str()),
                };
                ret.env
            });

        frames.push(frame);
        ret
    });

    let bytes_to_dynimg = move |bytes: Vec<u8>| -> image::RgbaImage {
        image::ImageBuffer::from_raw(width, height, bytes).unwrap()
    };

    let mut out_path = String::new();

    out_path.push_str(
        std::path::Path::new(&script_name)
            .file_stem()
            .and_then(|x| x.to_str())
            .expect("Could not extract file stem from script name"),
    );

    out_path.push('_');
    out_path.push_str(&img_path);

    match frame_count {
        0 => panic!("YO! What the FUCK am i supposed to do with 0 frames?"),
        1 => bytes_to_dynimg(frames.pop().unwrap()).save(&out_path)?,
        _ => {
            out_path.push_str(".gif");
            use image::codecs::gif::*;
            use std::fs::File;
            let mut encoder = GifEncoder::new(File::create(&out_path)?);
            let delay = image::Delay::from_numer_denom_ms(1000, 10);
            encoder.set_repeat(Repeat::Infinite)?;
            encoder.encode_frames(
                frames
                    .into_iter()
                    .map(bytes_to_dynimg)
                    .map(|img| image::Frame::from_parts(img, 0, 0, delay)),
            )?;
        }
    }

    println!("Output to: {}", out_path);

    Ok(())
}
