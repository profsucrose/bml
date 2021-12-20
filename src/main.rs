use image::io::Reader as ImageReader;
mod ast;
use ast::Val;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut args: Vec<String> = std::env::args().skip(1).collect();
  print!("\n{}\n", args.join(", "));
  let (script, img_path): (&dyn Fn() -> (ast::Env, ast::Ast), _) = {
    match (args.pop(), args.pop(), args.pop()) {
      (Some(img), Some(script), None) => match script.as_str() {
        "invert_color" => (&ast::examples::invert_color, img),
        "closing_circle" => (&ast::examples::closing_circle, img),
        _ => panic!("No script named {}", script),
      }
      _ => panic!("Found {} arguments, expected {}", args.len() + 3, 2),
    }
  };

  let mut image = ImageReader::open(img_path.as_str())?.decode()?;
  let buffer = image.as_mut_rgba8().expect("Couldn't get pixel buffer from image");

  let (mut env, ast) = script();
  let (width, height) = buffer.dimensions();
  env.set(2, Val::Vec2(width as f32, height as f32));
  env.set(5, Val::Float(1.0)); /* max frame */
  buffer.enumerate_pixels_mut().fold(env, |mut env, (x, y, image::Rgba([r, g, b, a]))| {
    env.set(3, Val::Vec2(x as f32, y as f32));
    env.set(4, Val::Vec4(
      *r as f32 / 255.0,
      *g as f32 / 255.0,
      *b as f32 / 255.0,
      *a as f32 / 255.0,
    ));
    env.set(6, Val::Float(0.0)); /* frame */
    let ret = ast::eval(&ast, env);
    match ret.env.ret {
      Some(Val::Vec4(x, y, z, w)) => {
        *r = (255.0 * x) as u8;
        *g = (255.0 * y) as u8;
        *b = (255.0 * z) as u8;
        *a = (255.0 * w) as u8;
      },
      None => panic!("scripts must return new pixel color"),
      Some(_) => panic!("pixel colors returned must be 4 dimensional"),
    };
    ret.env
  });

  let mut out_path = "corrupted_".to_string();
  out_path.push_str(img_path.as_str());
  image.save(out_path.as_str())?;
  println!("Output to: {}", out_path);

  Ok(())
}
