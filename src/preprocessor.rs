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
                    let name = self.match_token(TokenType::Identifier);

                    if name.is_none() {
                        panic!("[line {}] Error: Expected macro definition", token.line);
                    }

                    let name = name.unwrap();

                    if self.match_token(TokenType::LeftParen).is_none() {
                        panic!("[line {}] Error: Expected closing parenthesis", token.line);
                    }

                    let mut parameters = Vec::new();

                    loop {
                        // read parameters until closing parenthesis
                        if self.match_token(TokenType::RightParen).is_some() {
                            break;
                        }

                        let parameter = self.match_token(TokenType::Identifier);

                        println!("param: {:?}", parameter);

                        if parameter.is_none() {
                            panic!(
                                "[line {}] Error: Expected parameter in macro definition",
                                token.line
                            );
                        }

                        let parameter = parameter.unwrap();

                        parameters.push(parameter.lexeme);

                        self.match_token(TokenType::Comma);
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
                            loop {
                                let t = self.advance();

                                if t.token_type == TokenType::RightParen {
                                    break;
                                }
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
                        loop {
                            let t = self.advance();

                            if t.token_type == TokenType::RightParen {
                                break;
                            }
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
        let mut cursor = 0;

        println!("Expanding macro: {:?}", call);

        let mut args = Vec::new();
        let mut current_arg = Vec::new();

        let name = call.get(cursor);
        cursor += 1;

        if name.is_none() {
            // TODO: handle error
        }

        let name = name.unwrap();

        // consume left paren
        if call.get(cursor).is_none() {
            // TODO: handle error
        } else {
            cursor += 1;
        }

        loop {
            let token = call.get(cursor);
            cursor += 1;

            if token.is_none() {
                // TODO: throw error
                // expected closing paren
            }

            let token = token.unwrap();

            match token.token_type {
                TokenType::Comma => {
                    if current_arg.is_empty() {
                        // TODO: throw error
                    }

                    args.push(current_arg.clone());
                    current_arg.clear();
                }
                TokenType::RightParen => {
                    if !current_arg.is_empty() {
                        args.push(current_arg.clone());
                        drop(current_arg); // should no longer be used
                    }

                    break;
                }

                // expand nested macro call
                TokenType::Identifier => {
                    if self.macros.contains_key(&token.lexeme) {
                        let start = cursor;

                        // TODO: make sure call is valid
                        loop {
                            cursor = cursor + 1;

                            if call.get(cursor).unwrap().token_type == TokenType::RightParen {
                                break;
                            }
                        }

                        let nested_call = call[start..cursor].to_owned();
                        let mut expansion = self.expand_macro(token.line, &nested_call);

                        current_arg.append(&mut expansion);
                    }
                }

                _ => {
                    current_arg.push(token.to_owned());
                }
            }
        }

        self.macros.get(&name.lexeme).unwrap().expand(line, args)
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

    fn match_token(&mut self, token_type: TokenType) -> Option<Token> {
        let token = self.peek();
        if token.token_type == token_type {
            self.current += 1;
            return Some(token);
        }

        None
    }
}
