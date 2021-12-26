use std::process;

use colored::Colorize;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug)]
pub enum ErrorType {
    Parse,
    Preprocessor,
    Runtime,
    Scanner
}

pub fn info(message: &str) {
    println!("{}", message);
}

pub fn success_eval<S: AsRef<str>>(output: Option<S>) {
    print!("{}", "Returned: ".bold());

    match output {
        Some(output) => print!("{}", output.as_ref()),
        None => print!("<null>")
    }

    println!();
}

pub fn success_image<S: AsRef<str>>(path: S) {
    print!("{} ", "Success!".green().bold());
    print!("Output to: ");
    print!("{}", path.as_ref().underline());
    println!();
}

pub fn help() -> ! {
    println!("{}", format!("bml v{}", VERSION).bold());

    println!();

    println!(" Usage: 
  bml eval <script>
      Evaluate script and print the returned result.
  bml process <script> <image> <frames> [output_path]
      Process and manipulate an image with a script and output the result, either at an automatically determined path or at a specified path.
  bml new <script> <width> <height> <frames> <output_path>
      Generate new image with a given width and height with a script and write the result to a path.");
    
      println!();
    
    println!(" Process and manipulate images, or generate new ones, using the Buffer Manipulation Language.");

    process::exit(1);
}

pub fn report<S: AsRef<str>>(error: ErrorType, line: usize, message: S) -> ! {
    println!(
        "{}{}{}",
        format!("{:?}Error", error).bright_red().bold(),
        format!(" [line {}]", line).yellow(),
        format!(": {}", message.as_ref())
    );

    println!("{}", "Aborting...".bold());

    process::exit(1)
}
