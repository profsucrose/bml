use std::collections::HashMap;

use crate::{token::Token, token_type::TokenType, r#macro::Macro};

pub struct PreProcessor<'a> {
    tokens: &'a Vec<Token>,
    result: Vec<Token>,
    current: usize
}

impl<'a> PreProcessor<'a> {
    pub fn from(tokens: &Vec<Token>) -> PreProcessor {
        PreProcessor { tokens, result: vec![], current: 0 }
    }

    pub fn process(&mut self) -> &Vec<Token> {
        let mut macros = HashMap::new();

        loop {
            if self.at_end() { break }

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
                            break
                        }

                        let parameter = self.match_token(TokenType::Identifier);

                        println!("param: {:?}", parameter);

                        if parameter.is_none() {
                            panic!("[line {}] Error: Expected parameter in macro definition", token.line);
                        }

                        let parameter = parameter.unwrap();

                        parameters.push(parameter.lexeme);

                        self.match_token(TokenType::Comma);
                    }

                    let mut template = Vec::new();

                    // read to rest of line for macro template expansion
                    loop {
                        if self.at_end() || self.peek().line != token.line {
                            break
                        }

                        let template_token = self.advance();
                        template.push((template_token.token_type, template_token.lexeme));
                    }

                    macros.insert(name.lexeme, Macro::new(parameters, template));
                },

                // macro calls
                TokenType::Identifier => { 
                    if self.peek().token_type == TokenType::LeftParen {
                        self.advance();

                        let mut args = Vec::new();

                        'outer: loop {
                            let mut arg = Vec::new();

                            loop {
                                if self.match_token(TokenType::Comma).is_some() { break }
                                if self.match_token(TokenType::RightParen).is_some() {
                                    if !arg.is_empty() { args.push(arg); }
                                    break 'outer;
                                }

                                arg.push(self.advance());
                            }

                            println!("pushed arg: {:?}", arg);
                            args.push(arg);
                        }

                        println!("ended call read: {}", self.peek().lexeme);

                        let call = macros.get(&token.lexeme);

                        if call.is_none() {
                            panic!("[line {}] Error: macro '{}' is undefined", token.line, token.lexeme);
                        }

                        println!("args: {:?}", args);
                        let expansion = call.unwrap().expand(token.line, args);

                        for token in expansion {
                            self.result.push(token);
                        }
                        
                    } else {
                        self.result.push(token);
                    }
                },

                _ => {
                    println!("Pushing {:?}", token.lexeme);
                    self.result.push(token);
                }
            }
        }

        &self.result
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
            return Some(token)
        }

        None
    }
}