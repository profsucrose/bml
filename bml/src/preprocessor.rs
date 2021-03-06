use std::collections::HashMap;

use crate::{
    logger::{report, ErrorType},
    r#macro::Macro,
    token::Token,
    token_type::TokenType,
};

#[derive(Default)]
pub struct Arg {
    parens: usize,
    square_brackets: usize,
    brackets: usize,
    pub lexeme: Vec<Token>,
}

impl Arg {
    pub fn is_closed(&self) -> bool {
        self.parens == 0 && self.brackets == 0 && self.square_brackets == 0
    }

    pub fn new_token(&mut self, token: Token) -> Result<(), &str> {
        match token.token_type {
            TokenType::LeftParen => self.parens += 1,
            TokenType::RightParen => {
                if self.parens == 0 {
                    return Err("Unexpected ')' in macro arguments");
                } else {
                    self.parens -= 1
                }
            },
            TokenType::LeftBracket => self.brackets += 1,
            TokenType::RightBracket => {
                if self.brackets == 0 {
                    return Err("Unexpected '}' in macro arguments");
                } else {
                    self.brackets -= 1
                }
            },
            TokenType::LeftSquare => self.square_brackets += 1,
            TokenType::RightSquare => {
                if self.square_brackets == 0 {
                    return Err("Unexpected ']' in macro arguments");
                } else {
                    self.square_brackets -= 1
                }
            },
            _ => {}
        }

        self.lexeme.push(token);

        Ok(())
    }
}

pub struct PreProcessor {
    tokens: Vec<Token>,
    result: Vec<Token>,
    current: usize,
    macros: HashMap<String, Macro>,
}

impl PreProcessor {
    pub fn from(tokens: Vec<Token>) -> PreProcessor {
        PreProcessor {
            tokens,
            result: vec![],
            current: 0,
            macros: HashMap::new(),
        }
    }

    pub fn process(mut self) -> Vec<Token> {
        while !self.at_end() {
            let token = self.advance();

            match token.token_type {
                // macro definition
                TokenType::Macro => {
                    let name = self.consume(TokenType::Identifier);

                    if name.is_none() {
                        report(
                            ErrorType::Preprocessor,
                            token.line,
                            "Expected macro definition",
                        );
                    }

                    let name = name.unwrap();

                    if self.consume(TokenType::LeftParen).is_none() {
                        report(
                            ErrorType::Preprocessor,
                            token.line,
                            "Expected opening parenthesis",
                        );
                    }

                    // read parameters
                    let mut parameters = Vec::new();

                    while self.consume(TokenType::RightParen).is_none() {
                        let parameter = self.consume(TokenType::Identifier);

                        if parameter.is_none() {
                            report(
                                ErrorType::Preprocessor,
                                token.line,
                                "Expected parameter in macro definition",
                            );
                        }

                        parameters.push(parameter.unwrap().lexeme);

                        self.consume(TokenType::Comma);
                    }

                    let mut template = Vec::new();

                    // read to rest of line for macro template expansion

                    let mut brackets = 0;

                    while !self.at_end() && (self.peek().line == token.line || brackets > 0) {
                        let template_token = self.advance();

                        match template_token.token_type {
                            TokenType::LeftBracket => brackets += 1,
                            TokenType::RightBracket => {
                                if brackets == 0 {
                                    report(
                                        ErrorType::Preprocessor,
                                        self.peek().line,
                                        "Unexpected '}'",
                                    )
                                }

                                brackets -= 1
                            }
                            _ => {}
                        };

                        if template_token.token_type == TokenType::Identifier
                            && self.peek().token_type == TokenType::LeftParen
                            && self.macros.contains_key(&template_token.lexeme)
                        {
                            let start = self.current - 1;
                            self.consume(TokenType::LeftParen);
                            let mut parens = 1;

                            while !self.at_end() && parens > 0 {
                                let t = self.advance();

                                match t.token_type {
                                    TokenType::LeftParen => parens += 1,
                                    TokenType::RightParen => parens -= 1,
                                    TokenType::RightBracket => brackets += 1,
                                    TokenType::LeftBracket => brackets -= 1,
                                    _ => {}
                                };
                            }

                            if parens > 0 {
                                report(
                                    ErrorType::Preprocessor,
                                    self.previous().line,
                                    "Expected ')' in macro definition",
                                );
                            }

                            let nested_call = self.tokens[start..self.current].to_owned();
                            let mut expansion = self.expand_macro(token.line, &nested_call);

                            template.append(&mut expansion)
                        } else {
                            template.push(template_token)
                        }
                    }

                    if brackets > 0 {
                        report(
                            ErrorType::Preprocessor,
                            self.peek().line,
                            format!("Expected '}}' in '{}' macro definition", name.lexeme),
                        );
                    }

                    self.macros
                        .insert(name.lexeme, Macro::new(parameters, template));
                }

                // macro calls
                TokenType::Identifier => {
                    if self.peek().token_type == TokenType::LeftParen
                        && self.macros.contains_key(&token.lexeme)
                    {
                        let start = self.current - 1;

                        self.consume(TokenType::LeftParen);

                        let mut parens = 1;

                        while !self.at_end() && parens > 0 {
                            match self.advance().token_type {
                                TokenType::LeftParen => parens += 1,
                                TokenType::RightParen => parens -= 1,
                                _ => {}
                            }
                        }

                        // if reached EoF w/o parens == 0
                        if parens != 0 {
                            report(
                                ErrorType::Preprocessor,
                                self.previous().line,
                                "Missing ')' when parsing macro call",
                            );
                        }

                        let nested_call = self.tokens[start..self.current].to_owned();
                        let mut expansion = self.expand_macro(token.line, &nested_call);

                        self.result.append(&mut expansion);
                    } else {
                        self.result.push(token);
                    }
                }

                _ => {
                    self.result.push(token);
                }
            }
        }

        self.result
    }

    // dot([expr], [expr]) -> expansion
    fn expand_macro(&self, line: usize, call: &Vec<Token>) -> Vec<Token> {
        // assert that call starts with identifier, has left paren,
        let name = match call.get(0) {
            Some(token) => {
                if token.token_type != TokenType::Identifier {
                    report(
                        ErrorType::Preprocessor,
                        token.line,
                        format!(
                            "Expected macro call to start with identifier, got '{:?}'",
                            token.lexeme
                        )
                        .as_str(),
                    );
                }

                token
            }
            None => {
                report(
                    ErrorType::Preprocessor,
                    line,
                    "Expected macro call to have non-zero length",
                );
            }
        };

        match call.get(1) {
            Some(Token {
                token_type: TokenType::LeftParen,
                ..
            }) => {}
            _ => {
                report(
                    ErrorType::Preprocessor,
                    line,
                    "Expected '(' in macro invocation",
                );
            }
        }

        let mut cursor = 2;
        let mut current_arg = Arg::default();
        let mut args = Vec::new();

        loop {
            match call.get(cursor) {
                None => {
                    report(ErrorType::Preprocessor, line, "Expected ')' in macro call");
                }
                Some(token) => {
                    let mut is_arg_comma = false;

                    match token.token_type {
                        TokenType::Comma => {
                            if current_arg.is_closed() {
                                if current_arg.lexeme.is_empty() {
                                    report(ErrorType::Preprocessor, line, "Unexpected ',' in macro arguments");
                                }

                                args.push(current_arg.lexeme);
                                current_arg = Arg::default();

                                is_arg_comma = true;
                            }
                        }
                        TokenType::RightParen => {
                            if current_arg.is_closed() {
                                if !current_arg.lexeme.is_empty() {
                                    args.push(current_arg.lexeme);
                                }

                                break;
                            }
                        }
                        TokenType::Identifier => {
                            if self.macros.contains_key(&token.lexeme) {
                                let start = cursor;

                                loop {
                                    match call.get(cursor) {
                                        Some(Token {
                                            token_type: TokenType::RightParen,
                                            ..
                                        }) => {
                                            cursor += 1;
                                            break;
                                        }
                                        None => {
                                            report(
                                                ErrorType::Preprocessor,
                                                line,
                                                "Expected closing ')' in macro call",
                                            );
                                        }
                                        _ => cursor += 1,
                                    }
                                }

                                let nested_call = call[start..cursor].to_owned();
                                let mut expansion = self.expand_macro(token.line, &nested_call);

                                current_arg.lexeme.append(&mut expansion);

                                // cursor is already incremented
                                continue;
                            }
                        }
                        _ => {}
                    }

                    if !is_arg_comma {
                        if let Err(error) = current_arg.new_token(token.clone()) {
                            report(ErrorType::Preprocessor, line, error);
                        }
                    }
               }
            }

            cursor += 1;
        }

        self.macros.get(&name.lexeme).unwrap().expand(args)
    }

    fn at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }

    fn advance(&mut self) -> Token {
        let token = self.peek();
        self.current += 1;
        token
    }

    fn previous(&self) -> Token {
        if self.current == 0 {
            panic!("Tried to call previous() at start of file");
        }

        self.tokens.get(self.current - 1).unwrap().to_owned()
    }

    fn peek(&self) -> Token {
        if self.at_end() {
            panic!("Error: unexpected EOF");
        }

        self.tokens.get(self.current).unwrap().to_owned()
    }

    fn consume(&mut self, token_type: TokenType) -> Option<Token> {
        let token = self.peek();
        if token.token_type == token_type {
            self.current += 1;
            return Some(token);
        }

        None
    }
}
