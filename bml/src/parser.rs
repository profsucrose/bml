use std::collections::HashMap;

use lasso::{Rodeo, Spur};

use crate::ast::{Ast, BuiltIn, Field, Op, Swizzle, Val, AstNode};
use crate::logger::{report, ErrorType};
use crate::token::Token;
use crate::token_type::TokenType;

/*
    precedence rules:

    stmt: expr if expr else expr | assign | "return" expr
    assign: identifier "=" expr
    builtin: dist
    arguments: argument ("," argument)*
    argument: expr
    expr: equality | block | call
    block: "{" blockStmt+ "}"
    blockStmt: stmt | "give" expr
    equality: comparison ("==" comparison)*
    comparison: term (("<" | ">" | "<=" | ">=") term)*
    term: factor (("+" | "-") factor)*
    factor: access (("*" | "/") access)*
    access: primary ("." ("x"|"y"|"z"|"w")*)
    primary: identifier | literal | "(" expression ")" | access | call
    call: builtin "(" arguments? ")"
*/

pub struct Parser<'a> {
    tokens: &'a Vec<Token>,
    current: usize,
    pub rodeo: Rodeo<Spur>,
    builtins: HashMap<String, BuiltIn>,
    lines: Vec<Ast>,
}

impl<'a> Parser<'a> {
    pub fn from(tokens: &Vec<Token>) -> Parser {
        let mut builtins = HashMap::new();

        builtins.insert(String::from("dist"), BuiltIn::Dist);
        builtins.insert(String::from("radians"), BuiltIn::Radians);
        builtins.insert(String::from("pow"), BuiltIn::Pow);
        builtins.insert(String::from("sin"), BuiltIn::Sin);

        Parser {
            tokens,
            current: 0,
            rodeo: Rodeo::new(),
            builtins,
            lines: Vec::new(),
        }
    }

    pub fn parse(mut self) -> (Rodeo, Ast) {
        while !self.at_end() {
            let statement = self.statement();

            self.lines.push(statement);
        }

        let prev = self.previous().line;
        let Self { rodeo, lines, .. } = self;
        (rodeo, Ast::new(AstNode::Block(lines), prev))
    }

    fn statement(&mut self) -> Ast {
        if self.match_token(TokenType::Identifier) {
            return self.assign();
        }
        if self.match_token(TokenType::Return) {
            return self.r#return();
        }

        self.r#if()
    }

    fn r#if(&mut self) -> Ast {
        let mut expression = self.expression();

        if self.match_token(TokenType::If) {
            let condition = self.expression();

            self.consume(TokenType::Else, "Expected 'else' in if expression");

            let false_ret = self.expression();

            expression = Ast::new(
                AstNode::If {
                    cond: Box::new(condition),
                    true_ret: Box::new(expression),
                    false_ret: Box::new(false_ret),
                },
                self.previous().line
            );
        }

        Ast::new(expression.node, self.previous().line)
    }

    fn r#return(&mut self) -> Ast {
        let expression = self.r#if();

        Ast::new(AstNode::Return(Box::new(expression)), self.previous().line)
    }

    fn assign(&mut self) -> Ast {
        let name = self.previous().lexeme;

        self.consume(TokenType::Equals, "Expected '=' after variable declaration");

        let initializer = self.r#if();

        Ast::new(AstNode::Assign(self.identifier(name), Box::new(initializer)), self.previous().line)
    }

    fn expression(&mut self) -> Ast {
        if self.match_token(TokenType::LeftBracket) { 
            return self.block(); 
        }

        self.equality()
    }

    fn equality(&mut self) -> Ast {
        let mut equality = self.comparison();

        while self.match_token(TokenType::EqualsEquals) {
            let operator = match self.previous().token_type {
                TokenType::EqualsEquals => Op::Equal,
                _ => report(
                    ErrorType::Parse,
                    self.previous().line,
                    format!("Unexpected binary operator \'{}\'", self.previous().lexeme).as_str(),
                ),
            };

            let right = self.comparison();

            equality = Ast::new(AstNode::BinOp(Box::new(equality), operator, Box::new(right)), self.previous().line);
        }

        equality
    }

    fn access(&mut self) -> Ast {
        let mut access = self.primary();

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

            access = Ast::new(AstNode::VecAccess(
                Box::new(access),
                Swizzle(
                    *accessors.get(0).unwrap(),
                    accessors.get(1).copied(),
                    accessors.get(2).copied(),
                    accessors.get(3).copied(),
                ),
            ), self.previous().line);
        }

        access
    }

    fn comparison(&mut self) -> Ast {
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

            term = Ast::new(AstNode::BinOp(Box::new(term), op, Box::new(right)), self.previous().line);
        }

        term
    }

    fn term(&mut self) -> Ast {
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

            factor = Ast::new(AstNode::BinOp(Box::new(factor), op, Box::new(right)), self.previous().line);
        }

        factor
    }

    fn factor(&mut self) -> Ast {
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

            primary = Ast::new(AstNode::BinOp(Box::new(primary), op, Box::new(right)), self.previous().line);
        }

        primary
    }

    fn primary(&mut self) -> Ast {
        if self.peek().token_type == TokenType::Identifier
            && self.peek_next().token_type == TokenType::LeftParen
        {
            self.current += 1;
            return self.call();
        }

        if self.match_token(TokenType::LeftParen) {
            let expression = self.r#if();

            self.consume(TokenType::RightParen, "Expect ')' after expression");

            // handle parenthesis
            return Ast::new(AstNode::Block(vec![Ast::new(AstNode::Give(Box::new(expression)), self.previous().line)]), self.previous().line);
        }

        if self.match_token(TokenType::Identifier) {
            return self.identity();
        }

        self.literal()
    }

    fn literal(&mut self) -> Ast {
        // number literal
        if self.match_token(TokenType::Number) {
            return self.number();
        }

        if self.match_token(TokenType::LeftSquare) {
            return self.vector();
        }

        report(ErrorType::Parse, self.peek().line, format!("Unexpected '{}' in expression", self.peek().lexeme).as_str())
    }

    fn vector(&mut self) -> Ast {

        let first = self.r#if();

        if self.match_token(TokenType::Semi) {
            self.consume(TokenType::Number, "Expected scalar literal as length in vector");

            let length = self.number();

            self.consume(
                TokenType::RightSquare,
                "vector literals must end with ']' and be up to 4 components",
            );

            return Ast::new(AstNode::VecRepeated(Box::new(first), Box::new(length)), self.previous().line);
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

        Ast::new(AstNode::VecLiteral(Box::new(first), Box::new(second), third, fourth), self.previous().line)
    }

    fn number(&mut self) -> Ast {
        Ast::new(AstNode::V(Val::Float(self.previous().lexeme.parse::<f32>().unwrap())), self.previous().line)
    }

    fn identity(&mut self) -> Ast {
        Ast::new(AstNode::Ident(self.rodeo.get_or_intern(self.previous().lexeme)), self.previous().line)
    }

    fn call(&mut self) -> Ast {
        let token = self.previous();
        let name = &token.lexeme;

        let builtin = if let Some(&builtin) = self.builtins.get(name) {
            builtin
        } else {
            report(ErrorType::Parse, token.line, format!("Built-in function '{}' does not exist", name).as_str())
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

        Ast::new(AstNode::Call(builtin, arguments), self.previous().line)
    }

    fn block(&mut self) -> Ast {
        let mut statements = Vec::new();

        while !self.match_token(TokenType::RightBracket) {
            if self.match_token(TokenType::Give) {
                statements.push(self.give());
            } else {
                statements.push(self.statement());
            }
        }

        Ast::new(AstNode::Block(statements), self.previous().line)
    }

    fn give(&mut self) -> Ast {
        let expression = self.r#if();

        Ast::new(AstNode::Give(Box::new(expression)), self.previous().line)
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
        // TODO: clean error
        self.tokens.get(self.current).expect("Unexpected EOF")
    }
}
