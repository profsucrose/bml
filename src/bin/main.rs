use image::io::Reader as ImageReader;
use bml::ast;
use ast::Val;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args: Vec<String> = std::env::args().skip(1).collect();
  let img_path = args.pop().expect("expected image path");
  let script_name = args.pop().expect("expected script name");
  if args.pop().is_some() { panic!("expected exactly two arguments") }
  let (frames, script): (_, &dyn Fn() -> ast::Ast) = {
    match script_name.as_str() {
      "invert_color" => (1, &ast::examples::invert_color),
      "closing_circle" => (20, &ast::examples::closing_circle),
      _ => panic!("No script named {}", script_name),
    }
  };

  let mut image = ImageReader::open(&img_path)?.decode()?;
  let buffer = image.as_mut_rgba8().expect("Couldn't get pixel buffer from image");

  let (mut env, ast) = (ast::Env::default(), script());
  let (width, height) = buffer.dimensions();
  env.set(2, Val::Vec2(width as _, height as _));
  env.set(5, Val::Float(frames as _));
  let mut framevec = Vec::with_capacity(frames);
  (0..frames).fold(env, |mut env, frame| {
    env.set(6, Val::Float(frame as _));

    let mut frame = Vec::<u8>::with_capacity((width * height * 4) as usize);
    let ret = buffer.enumerate_pixels().fold(env, |mut env, (x, y, rgba)| {
      let [r, g, b, a] = rgba.0;
      env.set(3, Val::Vec2(x as _, y as _));
      env.set(4, Val::Vec4(
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
      ));
      let mut ret = ast::eval(&ast, env);
      match ret.env.ret.take() {
        Some(Val::Vec4(x, y, z, w)) => {
          // println!("{:?}, {:?}", ret.env.get(2), ret.env.get(5));
          // println!("{:?}, {:?}", ret.env.get(5), ret.env.get(6));
          // println!("{}, {}, {}, {}", x, y, z, w);
          frame.push((255.0 * x) as u8);
          frame.push((255.0 * y) as u8);
          frame.push((255.0 * z) as u8);
          frame.push((255.0 * w) as u8);
        },
        None => panic!("scripts must return new pixel color"),
        Some(_) => panic!("pixel colors returned must be 4 dimensional"),
      };
      ret.env
    });
    framevec.push(frame);
    ret
  });

  let bytes_to_dynimg = move |bytes: Vec<u8>| -> image::RgbaImage { 
    image::ImageBuffer::from_raw(width, height, bytes).unwrap()
  };

  let mut out_path = String::new();
  out_path.push_str(&script_name);
  out_path.push('_');
  out_path.push_str(&img_path);
  match frames {
    0 => panic!("wtf i am supposed to do with 0 frames?"),
    1 => bytes_to_dynimg(framevec.pop().unwrap()).save(&out_path)?,
    _ => {
      out_path.push_str(".gif");
      use std::fs::File;
      use image::codecs::gif::*;
      let mut encoder = GifEncoder::new(File::create(&out_path)?);
      let delay = image::Delay::from_numer_denom_ms(1000, 10);
      encoder.set_repeat(Repeat::Infinite)?;
      encoder.encode_frames(
        framevec
          .into_iter()
          .map(bytes_to_dynimg)
          .map(|img| image::Frame::from_parts(img, 0, 0, delay))
      )?;
    }
  }
  println!("Output to: {}", out_path);

  Ok(())
}
