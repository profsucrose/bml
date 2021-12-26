pub mod ast;
pub mod logger;
pub mod r#macro;
pub mod parser;
pub mod preprocessor;
pub mod scanner;
pub mod token;
pub mod token_type;
pub use parser::Parser;
pub use preprocessor::PreProcessor;
pub use scanner::Scanner;

pub fn string_to_ast<S: AsRef<str>>(string: S) -> (lasso::Rodeo, ast::SrcAst) {
    let raw = Scanner::from(string.as_ref().to_owned()).scan();
    let expanded = PreProcessor::from(raw).process();

    let (ro, ast) = Parser::from(&expanded).parse();
    let const_ro = lasso::Rodeo::new();

    (
        ro,
        ast.map(&|sa| match ast::is_const(&sa) {
            true => ast::SrcAst {
                ast: ast::Ast::V(ast::eval(&sa, ast::Env::default(), &const_ro).val.unwrap()),
                line: sa.line,
            },
            _ => sa,
        }),
    )
}
