use ast::Val;
use bml::{
    ast::{self, eval, Sampler},
    logger::{help, report, success_eval, success_image, ErrorType, render_progress},
};
use image::{io::Reader as ImageReader, DynamicImage};
use rayon::prelude::*;
use std::{path::Path, process, sync::{Mutex, Arc}};

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

    let (script, image, output_path, width, height, frame_count) = match args.next().as_deref() {
        None => {
            help();
        }
        Some("eval") => {
            let script = if let Some(script) = args.next() {
                script
            } else {
                help();
            };

            let (rodeo, ast) = bml::string_to_ast(std::fs::read_to_string(script)?);
            let env = ast::Env::default();
            let output = eval(&ast, env, &rodeo).env.ret.take();

            success_eval(output.map(|v| format!("{:?}", v)));

            process::exit(0)
        }
        Some("process") => match (args.next(), args.next(), args.next(), args.next()) {
            (Some(script), Some(image), Some(frame_count), output) => {
                match frame_count.parse::<usize>() {
                    Ok(frame_count) => (script, Some(image), output, None, None, frame_count),
                    _ => help(),
                }
            }
            _ => help(),
        },
        Some("new") => match (
            args.next(),
            args.next(),
            args.next(),
            args.next(),
            args.next(),
        ) {
            (Some(script), Some(width), Some(height), Some(frame_count), Some(output)) => match (
                frame_count.parse::<usize>(),
                width.parse::<usize>(),
                height.parse::<usize>(),
            ) {
                (Ok(frames), Ok(width), Ok(height)) => (
                    script,
                    None,
                    Some(output),
                    Some(width),
                    Some(height),
                    frames,
                ),
                _ => help(),
            },
            _ => help(),
        },
        _ => help(),
    };

    // should be no additional arguments
    if args.next().is_some() {
        help();
    }

    let (mut rodeo, ast) = bml::string_to_ast(std::fs::read_to_string(&script)?);

    let buffer = match &image {
        Some(path) => ImageReader::open(path)?.decode()?.to_rgba8(),
        None => DynamicImage::new_rgba8(width.unwrap() as u32, height.unwrap() as u32).to_rgba8(),
    };

    let rti = RuntimeIdents::new(&mut rodeo);
    let mut env = ast::Env::with_sampler(Sampler::from(&buffer));

    let (width, height) = buffer.dimensions();

    env.set(rti.resolution, Val::Vec2(width as _, height as _));
    env.set(rti.frame_count, Val::Float(frame_count as _));

    let mut frames = Vec::with_capacity(frame_count);

    let thread_count = match std::env::var("BML_THREADS") {
        Ok(threads) => threads
            .parse::<usize>()
            .expect("Environment variable 'BML_THREADS' was not set to a positive integer"),
        Err(_) => 2,
    };

    let mut out_path = match &output_path {
        Some(output) => output.to_owned(),
        None => {
            let mut path = String::new();

            // examples/closing_circle.buf
            // examples/img/rocket.png

            // -> examples/img/closing_circle_rocket.png

            let image = image.as_ref().unwrap();

            let image_prefix = &image[0..image.rfind('/').map(|x| x + 1).unwrap_or(image.len())];

            path.push_str(image_prefix);

            let script_name = Path::new(&script)
                .file_stem()
                .and_then(|x| x.to_str())
                .expect("Could not extract stem from script path");

            path.push_str(script_name);

            let image_name = Path::new(image)
                .file_name()
                .and_then(|x| x.to_str())
                .expect("Could not extract file name from image path");
            let image_name = &image_name[0..image_name.rfind('.').unwrap_or(image_name.len())];

            path.push('_');
            path.push_str(image_name);

            path
        }
    };

    let progress = Arc::new(Mutex::new(0));

    (0..frame_count).for_each(|frame| {
        env.set(rti.frame, Val::Float(frame as _));

        let size = (width * height) as usize;

        let net_pixels = size * frame_count;

        let mut frame = (0..thread_count)
            .into_par_iter()
            .map(|i| {
                let result = (
                    i,
                    (size * i / thread_count..size * (i + 1) / thread_count)
                        .flat_map(|index| {
                            let x = (index as u32) % width;
                            let y = (index as u32) / width;

                            // println!("{} {} {}", index, x, y);
                            let [r, g, b, a] = buffer.get_pixel(x, y).0;

                            let mut env = env.clone();

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

                            let mut progress = progress.lock().unwrap();

                            *progress += 1;

                            if *progress % (net_pixels / 10) == 0 {
                                let percent = ((*progress as f32) / (net_pixels as f32) * 10.0) as usize;
                                render_progress(percent);
                            }

                            match ret.env.ret.take() {
                                Some(Val::Vec4(r, g, b, a)) => [
                                    (255.0 * r) as u8,
                                    (255.0 * g) as u8,
                                    (255.0 * b) as u8,
                                    (255.0 * a) as u8,
                                ],
                                None => report(
                                    ErrorType::Runtime,
                                    ast.line,
                                    "Expected program to return vec4, got nothing",
                                ),
                                Some(val) => report(
                                    ErrorType::Runtime,
                                    ast.line,
                                    format!("Returned color must be vec4, got {:?}", val),
                                ),
                            }
                        })
                        .collect::<Vec<u8>>(),
                );

                result
            })
            .collect::<Vec<(usize, Vec<u8>)>>();

        frame.sort_by(|(idx1, _), (idx2, _)| idx1.cmp(idx2));

        let frame = frame
            .into_iter()
            .flat_map(|(_, vec)| vec)
            .collect::<Vec<u8>>();

        frames.push(frame);
    });

    let bytes_to_dynimg = move |bytes: Vec<u8>| -> image::RgbaImage {
        image::ImageBuffer::from_raw(width, height, bytes).unwrap()
    };

    match frame_count {
        0 => panic!("Expected positive number of frames for image/gif generations, got 0"),
        1 => {
            if output_path.is_none() {
                out_path.push_str(".png");
            }

            bytes_to_dynimg(frames.pop().unwrap()).save(&out_path)?
        }
        _ => {
            if output_path.is_none() {
                out_path.push_str(".gif");
            }

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

    success_image(out_path.as_str());

    Ok(())
}
