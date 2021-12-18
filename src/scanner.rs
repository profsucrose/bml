use std::ops::Add;

use crate::{token::Token, token_type::TokenType};

pub struct Scanner {
    source: String,
    current: usize,
    line: usize,
    start: usize,
    tokens: Vec<Token>
}

impl Scanner {
    pub fn new(source: String) -> Scanner {
        Scanner { source, current: 0, line: 0, start: 0, tokens: vec![] }
    }

    pub fn scan(&mut self) -> &Vec<Token> {
        loop {
            println!("{} {}", self.current, self.source.len());
            if self.at_end() { break }
            self.scan_token();
        }

        self.tokens.push(Token::new(TokenType::Eof, String::new(), self.line));
        &self.tokens
    }

    fn scan_token(&mut self) {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenType::LeftParen),
            ')' => self.add_token(TokenType::RightParen),
            '{' => self.add_token(TokenType::LeftBracket),
            '}' => self.add_token(TokenType::RightBracket),
            ',' => self.add_token(TokenType::Comma),
            '.' => self.add_token(TokenType::Dot),
            '-' => self.add_token(TokenType::Minus),
            '+' => self.add_token(TokenType::Plus),
            ';' => self.add_token(TokenType::Semi),
            '*' => self.add_token(TokenType::Star),
            // '=' => self.add_token(self.match('=') ? )
            _ => {}
        }
    }

    fn add_token(&mut self, token_type: TokenType) {
        self.tokens.push(Token::new(token_type, self.source[self.start..self.current].to_string(), self.line));
    }

    fn advance(&mut self) -> char {
        let ch = self.source.chars().nth(self.current).unwrap();
        self.current += 1;
        ch
    }

    fn at_end(&self) -> bool {
        self.current >= self.source.len()
    }
}