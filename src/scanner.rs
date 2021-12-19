use std::{collections::HashMap, process};

use crate::{token::Token, token_type::TokenType};

pub struct Scanner {
    source: String,
    current: usize,
    line: usize,
    start: usize,
    tokens: Vec<Token>,
    keywords: HashMap<&'static str, TokenType>,
    swizzles: HashMap<char, TokenType>
}

impl Scanner {
    pub fn from(source: String) -> Scanner {
        let mut keywords = HashMap::new();

        keywords.insert("if", TokenType::If);
        keywords.insert("else", TokenType::Else);
        keywords.insert("repeat", TokenType::Repeat);
        keywords.insert("return", TokenType::Return);
        keywords.insert("give", TokenType::Give);
        keywords.insert("macro", TokenType::Macro);

        keywords.insert("vec2", TokenType::Vec2);
        keywords.insert("vec3", TokenType::Vec3);
        keywords.insert("vec4", TokenType::Vec4);

        let mut swizzles = HashMap::new();

        swizzles.insert('x', TokenType::X);
        swizzles.insert('y', TokenType::Y);
        swizzles.insert('z', TokenType::Z);
        swizzles.insert('w', TokenType::W);

        swizzles.insert('r', TokenType::X);
        swizzles.insert('g', TokenType::Y);
        swizzles.insert('b', TokenType::Z);
        swizzles.insert('a', TokenType::W);

        Scanner { 
            source, 
            current: 0, 
            line: 1, 
            start: 0, 
            tokens: vec![], 
            keywords,
            swizzles 
        }
    }

    pub fn scan(&mut self) -> &Vec<Token> {
        loop {
            if self.at_end() { break }
            self.start = self.current;
            self.scan_token();
        }

        self.tokens.push(Token::new(TokenType::Eof, String::new(), self.line));
        &self.tokens
    }

    fn scan_token(&mut self) {
        let c = self.advance();
        match c {
            '[' => self.add_token(TokenType::LeftSquare),
            ']' => self.add_token(TokenType::RightSquare),
            '(' => self.add_token(TokenType::LeftParen),
            ')' => self.add_token(TokenType::RightParen),
            '{' => self.add_token(TokenType::LeftBracket),
            '}' => self.add_token(TokenType::RightBracket),
            ';' => self.add_token(TokenType::Semi),
            ',' => self.add_token(TokenType::Comma),
            '-' => self.add_token(TokenType::Minus),
            '+' => self.add_token(TokenType::Plus),
            '*' => self.add_token(TokenType::Star),
            '/' => self.add_token(TokenType::Slash),
            '=' => {
                let token = if self.match_lexeme('=') { TokenType::EqualsEquals } else { TokenType::Equals };
                self.add_token(token);
            },
            '<' => {
                let token = if self.match_lexeme('=') { TokenType::LessThanEquals } else { TokenType::LessThan };
                self.add_token(token);
            },
            '>' => {
                let token = if self.match_lexeme('=') { TokenType::GreaterThanEquals } else { TokenType::GreaterThan };
                self.add_token(token);
            },
            '#' => {
                // single line comment, skip line
                loop {
                    self.advance();
                    if self.peek() == '\n' || self.at_end() { break; }
                }
            },
            ' ' | '\r' | '\t' => {},
            '\n' => self.line += 1,
            '.' => self.swazzle(),
            _ => {
                if c.is_digit(10) {
                    self.number();
                } else if c.is_alphabetic() {
                    self.identifier();
                } else {
                    // error
                    panic!("[line {}] Error: Unexpected character '{}'", self.line, c);
                    // process::exit(1);
                }
            }
        }
    }

    fn swazzle(&mut self) {
        self.add_token(TokenType::Dot);

        loop {
            if !self.at_end() {
                if let Some(&swazzler) = self.swizzles.get(&self.peek()) {
                    self.start = self.current;

                    self.advance();
                    self.add_token(swazzler);

                    continue;
                } 
            }

            break;
        }
    }

    fn identifier(&mut self) {
        loop {
            if self.at_end() || (!self.peek().is_alphanumeric() &&  self.peek() != '_') {
                break;
            }

            self.advance();
        }

        let text = self.source[self.start..self.current].to_string();
        if let Some(&keyword) = self.keywords.get(text.as_str()) {
            self.add_token(keyword);
        } else {
            self.add_token(TokenType::Identifier)
        }
    }

    fn number(&mut self) {
        self.consume_digits();

        if !self.at_end() && self.peek() == '.' && self.peek_next().is_digit(10) {
            self.advance();

            self.consume_digits();
        }

        self.add_token(TokenType::Number)
    }

    fn consume_digits(&mut self) {
        loop { 
            if !self.peek().is_digit(10) { 
                break;
            } 

            self.advance(); 
        }
    }

    fn peek_next(&self) -> char {
        self.char_at(self.current + 1)
    }

    fn peek(&self) -> char {
        self.char_at(self.current)
    }

    fn match_lexeme(&mut self, expected: char) -> bool {
        if self.at_end() { return false; }
        if self.char_at(self.current) != expected { return false; }

        self.current += 1;
        true
    }

    fn add_token(&mut self, token_type: TokenType) {
        self.tokens.push(Token::new(token_type, self.source[self.start..self.current].to_string(), self.line));
    }

    fn char_at(&self, n: usize) -> char {
        self.source.chars().nth(n).unwrap()
    }

    fn advance(&mut self) -> char {
        let ch = self.char_at(self.current);
        self.current += 1;
        ch
    }

    fn at_end(&self) -> bool {
        self.current >= self.source.len()
    }
}