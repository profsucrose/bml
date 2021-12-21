use std::collections::HashMap;

use crate::{r#macro::Macro, token::Token, token_type::TokenType};

// TODO: replace w/ string interning
pub struct PreProcessor<'a> {
    tokens: &'a Vec<Token>,
    result: Vec<Token>,
    current: usize,
    macros: HashMap<String, Macro>,
}

impl<'a> PreProcessor<'a> {
    pub fn from(tokens: &Vec<Token>) -> PreProcessor {
        PreProcessor {
            tokens,
            result: vec![],
            current: 0,
            macros: HashMap::new(),
        }
    }

    pub fn process(&mut self) -> &Vec<Token> {
        while !self.at_end() {
            let token = self.advance();

            match token.token_type {
                // macro definition
                TokenType::Macro => {
                    let name = self.consume(TokenType::Identifier);

                    if name.is_none() {
                        panic!("[line {}] Error: Expected macro definition", token.line);
                    }

                    let name = name.unwrap();

                    if self.consume(TokenType::LeftParen).is_none() {
                        panic!("[line {}] Error: Expected closing parenthesis", token.line);
                    }

                    // read parameters
                    let mut parameters = Vec::new();

                    while self.consume(TokenType::RightParen).is_none() {
                        let parameter = self.consume(TokenType::Identifier);

                        if parameter.is_none() {
                            panic!(
                                "[line {}] Error: Expected parameter in macro definition",
                                token.line
                            );
                        }

                        parameters.push(parameter.unwrap().lexeme);

                        self.consume(TokenType::Comma);
                    }

                    let mut template = Vec::new();

                    // read to rest of line for macro template expansion
                    while !self.at_end() && self.peek().line == token.line {
                        let template_token = self.advance();

                        if template_token.token_type == TokenType::Identifier
                            && self.peek().token_type == TokenType::LeftParen
                            && self.macros.contains_key(&template_token.lexeme)
                        {
                            let start = self.current - 1;

                            // TODO: make sure call is valid
                            while self.consume(TokenType::RightParen).is_none() {
                                self.advance();
                            }

                            let nested_call = self.tokens[start..self.current].to_owned();
                            let expansion = self.expand_macro(token.line, &nested_call);

                            for t in expansion {
                                template.push((t.token_type, t.lexeme));
                            }
                        } else {
                            template.push((template_token.token_type, template_token.lexeme));
                        }
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

                        // TODO: make sure call is valid
                        while self.consume(TokenType::RightParen).is_none() {
                            self.advance();
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

        &self.result
    }

    // dot([expr], [expr]) -> expansion
    fn expand_macro(&self, line: usize, call: &Vec<Token>) -> Vec<Token> {
        // assert that call starts with identifier, has left paren,
        // comma-delimited arguments, and then right paren
        println!("Expanding macro: {:?}", call);

        let name = match call.get(0) {
            Some(token) => {
                if token.token_type == TokenType::Identifier {
                    token
                } else {
                    panic!(
                        "[line {}] Error: expected macro call to start with identifier ({:?})",
                        line, call
                    );
                }
            }
            None => panic!(
                "[line {}] Error: expected macro call to have non-zero length: {:?}",
                line, call
            ),
        };

        match call.get(1) {
            Some(Token {
                token_type: TokenType::LeftParen,
                ..
            }) => {}
            _ => panic!("[line {}] Expected '(' in macro expansion call", line),
        }

        let mut cursor = 2;
        let mut current_arg = Vec::new();
        let mut args = Vec::new();

        loop {
            match call.get(cursor) {
                None => {
                    panic!("[line {}] Error: expected ')' in macro call", line);
                }
                Some(token) => match token {
                    Token {
                        token_type: TokenType::RightParen,
                        ..
                    } => {
                        println!("RIGHT PAREN: {:?}", current_arg);
                        if !current_arg.is_empty() {
                            // move current_arg as should no longer be used
                            args.push(current_arg);
                        }

                        break;
                    }
                    Token {
                        token_type: TokenType::Comma,
                        ..
                    } => {
                        args.push(current_arg.clone());
                        current_arg.clear();
                    }
                    Token {
                        token_type: TokenType::Identifier,
                        ..
                    } => {
                        if self.macros.contains_key(&token.lexeme) {
                            let start = cursor;

                            loop {
                                match call.get(cursor) {
                                    Some(Token {
                                        token_type: TokenType::RightParen,
                                        ..
                                    }) => break,
                                    None => panic!(
                                        "[line {}] Error: expected closing ')' in macro call",
                                        line
                                    ),
                                    _ => cursor += 1,
                                }
                            }

                            let nested_call = call[start..cursor].to_owned();
                            let mut expansion = self.expand_macro(token.line, &nested_call);

                            current_arg.append(&mut expansion);
                        } else {
                            current_arg.push(token.clone());
                        }
                    }
                    _ => {
                        current_arg.push(token.clone());
                    }
                },
            }

            cursor += 1;
        }

        println!(
            "Calling macro expansion with: {}, {}, {:?}",
            name.lexeme, line, args
        );
        return self.macros.get(&name.lexeme).unwrap().expand(line, args);
    }

    fn at_end(&self) -> bool {
        self.current >= self.tokens.len()
    }

    fn advance(&mut self) -> Token {
        let token = self.peek();
        self.current += 1;
        token
    }

    fn peek(&self) -> Token {
        if self.at_end() {
            panic!("Error: unexpected EOF");
        }

        self.tokens.get(self.current).unwrap().to_owned()
    }

    // TODO: DRYer error handling
    fn consume(&mut self, token_type: TokenType) -> Option<Token> {
        let token = self.peek();
        if token.token_type == token_type {
            self.current += 1;
            return Some(token);
        }

        None
    }
}
