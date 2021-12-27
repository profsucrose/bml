use std::collections::HashMap;

use lasso::{Rodeo, Spur};

use crate::ast::{Ast, BuiltIn, Field, Op, SrcAst, Swizzle, Val};
use crate::logger::{report, ErrorType};
use crate::token::Token;
use crate::token_type::TokenType;

pub struct Parser<'a> {
    tokens: &'a Vec<Token>,
    current: usize,
    pub rodeo: Rodeo<Spur>,
    builtins: HashMap<String, BuiltIn>,
    lines: Vec<SrcAst>,
}

macro_rules! builtins {
    ($( ($key:expr, $value:expr) ),*) => {{
        let mut map = HashMap::new();
        $( map.insert(String::from($key), $value); )*
        map
    }};
}

impl<'a> Parser<'a> {
    pub fn from(tokens: &Vec<Token>) -> Parser {
        let builtins = builtins!(
            ("sample", BuiltIn::Sample),
            ("dist", BuiltIn::Dist),
            ("radians", BuiltIn::Radians),
            ("degrees", BuiltIn::Degrees),
            ("sin", BuiltIn::Sin),
            ("cos", BuiltIn::Cos),
            ("tan", BuiltIn::Tan),
            ("asin", BuiltIn::Asin),
            ("acos", BuiltIn::Acos),
            ("atan", BuiltIn::Atan),
            ("pow", BuiltIn::Pow),
            ("exp", BuiltIn::Exp),
            ("log", BuiltIn::Log),
            ("sqrt", BuiltIn::Sqrt),
            ("invsqrt", BuiltIn::InverseSqrt),
            ("abs", BuiltIn::Abs),
            ("sign", BuiltIn::Sign),
            ("floor", BuiltIn::Floor),
            ("ceil", BuiltIn::Ceil),
            ("fract", BuiltIn::Fract),
            ("mod", BuiltIn::Mod),
            ("min", BuiltIn::Min),
            ("max", BuiltIn::Max),
            ("clamp", BuiltIn::Clamp),
            ("mix", BuiltIn::Mix),
            ("step", BuiltIn::Step),
            ("length", BuiltIn::Length),
            ("dot", BuiltIn::Dot),
            ("cross", BuiltIn::Cross),
            ("norm", BuiltIn::Norm),
            ("mat2", BuiltIn::Mat2),
            ("mat3", BuiltIn::Mat3),
            ("mat4", BuiltIn::Mat4),
            ("rotate_x", BuiltIn::RotateX),
            ("rotate_y", BuiltIn::RotateY),
            ("rotate_z", BuiltIn::RotateZ),
            ("rotate", BuiltIn::Rotate),
            ("scale", BuiltIn::Scale),
            ("translate", BuiltIn::Translate),
            ("ortho", BuiltIn::Ortho),
            ("lookat", BuiltIn::LookAt),
            ("perspective", BuiltIn::Perspective)
        );

        Parser {
            tokens,
            current: 0,
            rodeo: Rodeo::new(),
            builtins,
            lines: Vec::new(),
        }
    }

    pub fn parse(mut self) -> (Rodeo, SrcAst) {
        while !self.at_end() {
            let statement = self.statement();

            self.lines.push(statement);
        }

        let prev = if self.tokens.len() > 0 {
            0
        } else {
            self.previous().line
        };
        let Self { rodeo, lines, .. } = self;
        (rodeo, SrcAst::new(Ast::Block(lines), prev))
    }

    fn statement(&mut self) -> SrcAst {
        if self.match_token(TokenType::Identifier) {
            return self.assign();
        }

        if self.match_token(TokenType::Return) {
            return self.r#return();
        }

        if self.match_token(TokenType::Repeat) {
            return self.repeat();
        }

        if self.match_token(TokenType::While) {
            return self.r#while();
        }

        self.r#if()
    }

    fn r#while(&mut self) -> SrcAst {
        let cond = self.expression();

        self.consume(TokenType::Comma, "Expected ',' in while statement");

        if !self.match_token(TokenType::Number) {
            report(
                ErrorType::Parse,
                self.peek().line,
                format!(
                    "Expected positive number literal in while statement, got '{}'",
                    self.peek().lexeme
                )
                .as_str(),
            );
        }

        let times = match self.previous().lexeme.parse::<f32>() {
            Ok(times) => times,
            Err(_) => report(
                ErrorType::Parse,
                self.previous().line,
                format!(
                    "Expected number literal in while statement, got '{}'",
                    self.previous().lexeme
                )
                .as_str(),
            ),
        };

        if !self.match_token(TokenType::LeftBracket) {
            report(
                ErrorType::Parse,
                self.previous().line,
                format!(
                    "Expected '{{' in while block, got '{}'",
                    self.previous().lexeme
                )
                .as_str(),
            );
        }

        let block = self.block();

        SrcAst::new(Ast::While(Box::new(cond), times, Box::new(block)), self.previous().line)
    }

    fn repeat(&mut self) -> SrcAst {
        if !self.match_token(TokenType::Number) {
            report(
                ErrorType::Parse,
                self.peek().line,
                format!(
                    "Expected positive number literal in repeat statement, got '{}'",
                    self.peek().lexeme
                )
                .as_str(),
            );
        }

        let times = match self.previous().lexeme.parse::<f32>() {
            Ok(times) => times,
            Err(_) => report(
                ErrorType::Parse,
                self.previous().line,
                format!(
                    "Expected number literal in repeat statement, got '{}'",
                    self.previous().lexeme
                )
                .as_str(),
            ),
        };

        if !self.match_token(TokenType::LeftBracket) {
            report(
                ErrorType::Parse,
                self.previous().line,
                format!(
                    "Expected '{{' in repeat block, got '{}'",
                    self.previous().lexeme
                )
                .as_str(),
            );
        }

        let block = self.block();

        SrcAst::new(Ast::Repeat(times, Box::new(block)), self.previous().line)
    }

    fn r#if(&mut self) -> SrcAst {
        let mut expression = self.expression();

        if self.match_token(TokenType::If) {
            let condition = self.expression();

            self.consume(TokenType::Else, "Expected 'else' in if expression");

            let false_ret = self.expression();

            expression = SrcAst::new(
                Ast::If {
                    cond: Box::new(condition),
                    true_ret: Box::new(expression),
                    false_ret: Box::new(false_ret),
                },
                self.previous().line,
            );
        }

        SrcAst::new(expression.ast, self.previous().line)
    }

    fn r#return(&mut self) -> SrcAst {
        let expression = self.r#if();

        SrcAst::new(Ast::Return(Box::new(expression)), self.previous().line)
    }

    fn assign(&mut self) -> SrcAst {
        let name = self.previous().lexeme;

        self.consume(TokenType::Equals, "Expected '=' after variable declaration");

        let initializer = self.r#if();

        SrcAst::new(
            Ast::Assign(self.identifier(name), Box::new(initializer)),
            self.previous().line,
        )
    }

    fn expression(&mut self) -> SrcAst {
        self.equality()
    }

    fn equality(&mut self) -> SrcAst {
        let mut equality = self.comparison();

        while self.match_token(TokenType::EqualsEquals) 
            || self.match_token(TokenType::NotEquals)
        {
            let operator = match self.previous().token_type {
                TokenType::EqualsEquals => Op::Equal,
                TokenType::NotEquals => Op::NotEqual,
                _ => report(
                    ErrorType::Parse,
                    self.previous().line,
                    format!("Unexpected binary operator \'{}\'", self.previous().lexeme),
                ),
            };

            let right = self.comparison();

            equality = SrcAst::new(
                Ast::BinOp(Box::new(equality), operator, Box::new(right)),
                self.previous().line,
            );
        }

        equality
    }

    fn access(&mut self) -> SrcAst {
        let mut access = self.index();

        if self.match_token(TokenType::Dot) {
            let mut accessors = Vec::new();

            loop {
                let token = self.advance();

                match token.token_type {
                    TokenType::X => accessors.push(Field::X),
                    TokenType::Y => accessors.push(Field::Y),
                    TokenType::Z => accessors.push(Field::Z),
                    TokenType::W => accessors.push(Field::W),
                    _ => {
                        self.current -= 1;
                        break;
                    } // _ => panic!("[line {}] Error: Expected 'x', 'y', 'z', 'w', 'r', 'g', 'b', or 'a', got {}", token.line, token.lexeme)
                }
            }

            if accessors.len() < 1 {
                report(
                    ErrorType::Parse,
                    self.previous().line,
                    "Expected at least one component when swizzling",
                );
            }

            access = SrcAst::new(
                Ast::VecAccess(
                    Box::new(access),
                    Swizzle(
                        *accessors.get(0).unwrap(),
                        accessors.get(1).copied(),
                        accessors.get(2).copied(),
                        accessors.get(3).copied(),
                    ),
                ),
                self.previous().line,
            );
        }

        access
    }

    fn index(&mut self) -> SrcAst {
        let mut value = self.primary();

        if self.match_token(TokenType::LeftSquare) {
            let index = self.expression();

            self.consume(TokenType::RightSquare, "Expected ']' when indexing matrix");

            value = SrcAst::new(
                Ast::MatAccess(Box::new(value), Box::new(index)),
                self.previous().line,
            );
        }

        value
    }

    fn comparison(&mut self) -> SrcAst {
        let mut term = self.term();

        loop {
            let token = self.peek();

            let op = match token.token_type {
                TokenType::LessThan => Some(Op::Less),
                TokenType::GreaterThan => Some(Op::More),
                TokenType::LessThanEquals => Some(Op::LessEq),
                TokenType::GreaterThanEquals => Some(Op::MoreEq),
                _ => None,
            };

            if op.is_none() {
                break;
            }

            // advance cursor if comparison operand
            self.current += 1;

            let op = op.unwrap();

            let right = self.term();

            term = SrcAst::new(
                Ast::BinOp(Box::new(term), op, Box::new(right)),
                self.previous().line,
            );
        }

        term
    }

    fn term(&mut self) -> SrcAst {
        let mut factor = self.factor();

        loop {
            let token = self.peek();

            let op = match token.token_type {
                TokenType::Plus => Some(Op::Add),
                TokenType::Minus => Some(Op::Sub),
                _ => None,
            };

            if op.is_none() {
                break;
            }

            // advance cursor
            self.current += 1;

            let op = op.unwrap();

            let right = self.factor();

            factor = SrcAst::new(
                Ast::BinOp(Box::new(factor), op, Box::new(right)),
                self.previous().line,
            );
        }

        factor
    }

    fn factor(&mut self) -> SrcAst {
        let mut primary = self.access();

        loop {
            let token = self.peek();

            let op = match token.token_type {
                TokenType::Star => Some(Op::Mul),
                TokenType::Slash => Some(Op::Div),
                _ => None,
            };

            if op.is_none() {
                break;
            }

            // advance cursor
            self.current += 1;

            let op = op.unwrap();

            let right = self.access();

            primary = SrcAst::new(
                Ast::BinOp(Box::new(primary), op, Box::new(right)),
                self.previous().line,
            );
        }

        primary
    }

    fn primary(&mut self) -> SrcAst {
        if self.peek().token_type == TokenType::Identifier
            && self.peek_next().token_type == TokenType::LeftParen
        {
            self.current += 1;
            return self.call();
        }

        if self.match_token(TokenType::LeftParen) {
            let expression = self.r#if();

            self.consume(TokenType::RightParen, "Unclosed ')' in expression");

            // handle parenthesis
            return SrcAst::new(
                Ast::Block(vec![SrcAst::new(
                    Ast::Give(Box::new(expression)),
                    self.previous().line,
                )]),
                self.previous().line,
            );
        }

        if self.match_token(TokenType::LeftBracket) {
            return self.block();
        }

        if self.match_token(TokenType::Identifier) {
            return self.identity();
        }

        self.literal()
    }

    fn literal(&mut self) -> SrcAst {
        // number literal
        if self.check(TokenType::Minus) && self.check_next(TokenType::Number) {
            // consume - and number
            self.advance();
            self.advance();

            return self.number(false);
        }

        if self.match_token(TokenType::Number) {
            return self.number(true);
        }

        if self.match_token(TokenType::LeftSquare) {
            return self.vector();
        }

        report(
            ErrorType::Parse,
            self.peek().line,
            format!("Unexpected '{}' in expression", self.peek().lexeme),
        )
    }

    fn vector(&mut self) -> SrcAst {
        let first = self.r#if();

        if self.match_token(TokenType::Semi) {
            self.consume(
                TokenType::Number,
                "Expected scalar literal as length in vector",
            );

            let length = self.number(true);

            self.consume(
                TokenType::RightSquare,
                "vector literals must end with ']' and be up to 4 components",
            );

            return SrcAst::new(
                Ast::VecRepeated(Box::new(first), Box::new(length)),
                self.previous().line,
            );
        }

        self.consume(TokenType::Comma, "Expected ',' in vector arguments");

        let second = self.r#if();

        let third = if self.match_token(TokenType::Comma) {
            let expression = self.r#if();
            Some(Box::new(expression))
        } else {
            None
        };

        let fourth = if self.match_token(TokenType::Comma) {
            Some(Box::new(self.r#if()))
        } else {
            None
        };

        self.consume(
            TokenType::RightSquare,
            "vector literals must end with ']' and be up to 4 components",
        );

        SrcAst::new(
            Ast::VecLiteral(Box::new(first), Box::new(second), third, fourth),
            self.previous().line,
        )
    }

    fn number(&mut self, positive: bool) -> SrcAst {
        let sign = if positive { 1.0 } else { -1.0 };

        SrcAst::new(
            Ast::V(Val::Float(
                sign * self.previous().lexeme.parse::<f32>().unwrap(),
            )),
            self.previous().line,
        )
    }

    fn identity(&mut self) -> SrcAst {
        SrcAst::new(
            Ast::Ident(self.rodeo.get_or_intern(self.previous().lexeme)),
            self.previous().line,
        )
    }

    fn call(&mut self) -> SrcAst {
        let token = self.previous();
        let name = &token.lexeme;

        let builtin = if let Some(&builtin) = self.builtins.get(name) {
            builtin
        } else {
            report(
                ErrorType::Parse,
                token.line,
                format!("Built-in function '{}' does not exist", name),
            )
        };

        self.consume(
            TokenType::LeftParen,
            "Expected \'(\' in built-in function call",
        );

        let mut arguments = Vec::new();

        while !self.check(TokenType::RightParen) {
            let expression = self.r#if();

            self.match_token(TokenType::Comma);

            arguments.push(expression);
        }

        self.consume(
            TokenType::RightParen,
            "Expected \')\' in built-in function call",
        );

        SrcAst::new(Ast::Call(builtin, arguments), self.previous().line)
    }

    fn block(&mut self) -> SrcAst {
        let mut statements = Vec::new();

        while !self.match_token(TokenType::RightBracket) {
            if self.match_token(TokenType::Give) {
                statements.push(self.give());
            } else {
                statements.push(self.statement());
            }
        }

        SrcAst::new(Ast::Block(statements), self.previous().line)
    }

    fn give(&mut self) -> SrcAst {
        let expression = self.r#if();

        SrcAst::new(Ast::Give(Box::new(expression)), self.previous().line)
    }

    fn match_token(&mut self, token_type: TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            return true;
        }

        false
    }

    fn identifier(&mut self, name: String) -> Spur {
        self.rodeo.get_or_intern(name)
    }

    fn consume(&mut self, token_type: TokenType, error: &str) -> Token {
        if self.peek().token_type == token_type {
            return self.advance();
        }

        report(ErrorType::Parse, self.peek().line, error);
    }

    fn advance(&mut self) -> Token {
        if !self.at_end() {
            self.current += 1;
        }

        self.previous()
    }

    fn previous(&self) -> Token {
        self.tokens[self.current - 1].clone()
    }

    fn check_next(&self, token_type: TokenType) -> bool {
        self.current + 1 < self.tokens.len() && self.peek_next().token_type == token_type
    }

    fn check(&self, token_type: TokenType) -> bool {
        !self.at_end() && self.peek().token_type == token_type
    }

    fn at_end(&self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    fn peek_next(&self) -> &Token {
        self.tokens.get(self.current + 1).expect("Unexpected EOF")
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.current).expect("Unexpected EOF")
    }
}
