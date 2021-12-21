use std::collections::HashMap;

use lasso::{Rodeo, Spur};

use crate::ast::ast::{Ast, BuiltIn, Field, Op, Swizzle, Val};
use crate::token::Token;
use crate::token_type::TokenType;

// #[derive(Debug)]
// pub enum Ast {
//     V(Val),
//     Assign(Spur, Box<Ast>), /* i32 will prolly become lasso::Spur */
//     Block(Vec<Ast>),
//     VecLiteral(Box<Ast>, Box<Ast>, Option<Box<Ast>>, Option<Box<Ast>>),
//     VecAccess(Box<Ast>, Swizzle),
//     Ident(Spur), /* will prolly become lasso::Spur */
//     Return(Box<Ast>),
//     Give(Box<Ast>),
//     BinOp(Box<Ast>, Op, Box<Ast>),
//     If {
//         cond: Box<Ast>,
//         true_ret: Box<Ast>,
//         false_ret: Box<Ast>,
//     },
//     Call(BuiltIn, Vec<Ast>),
// }

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

const DEBUG: bool = true;

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

        Parser {
            tokens,
            current: 0,
            rodeo: Rodeo::new(),
            builtins,
            lines: Vec::new(),
        }
    }

    pub fn parse(&mut self) -> Ast {
        if DEBUG {
            println!("parse(); {:?}", self.peek())
        };
        while !self.at_end() {
            let statement = self.statement();

            self.lines.push(statement);
        }

        Ast::Block(self.lines.clone())
    }

    fn statement(&mut self) -> Ast {
        if DEBUG {
            println!("statement(); {:?}", self.peek())
        };
        if self.match_token(TokenType::Identifier) {
            return self.assign();
        }
        if self.match_token(TokenType::Return) {
            return self.r#return();
        }

        self.r#if()

        // TODO: handle error
        // panic!(
        //     "[line {}] Error: expected variable declaration or return statement",
        //     self.previous().line
        // )
    }

    fn r#if(&mut self) -> Ast {
        if DEBUG {
            println!("if(); {:?}", self.peek())
        };
        println!("If statement");

        let mut expression = self.expression();

        if self.match_token(TokenType::If) {
            let condition = self.expression();

            self.consume(TokenType::Else, "Expected 'else' in if expression");

            let false_ret = self.expression();

            expression = Ast::If {
                cond: Box::new(condition),
                true_ret: Box::new(expression),
                false_ret: Box::new(false_ret),
            }
        }

        expression
    }

    fn r#return(&mut self) -> Ast {
        if DEBUG {
            println!("return(); {:?}", self.peek())
        };
        let expression = self.r#if();

        Ast::Return(Box::new(expression))
    }

    fn assign(&mut self) -> Ast {
        if DEBUG {
            println!("assign(); {:?}", self.peek())
        };
        let name = self.previous().lexeme;

        self.consume(TokenType::Equals, "Expected '=' after variable declaration");

        let initializer = self.r#if();

        Ast::Assign(self.identifier(name), Box::new(initializer))
    }

    fn expression(&mut self) -> Ast {
        if DEBUG {
            println!("expression(); {:?}", self.peek())
        };

        if self.match_token(TokenType::LeftBracket) {
            println!("Reading block: {:?}", self.peek());
            return self.block();
        }

        self.equality()
    }

    fn equality(&mut self) -> Ast {
        if DEBUG {
            println!("equality(); {:?}", self.peek())
        };
        let mut equality = self.comparison();

        // TODO: add !=
        while self.match_token(TokenType::EqualsEquals) {
            let operator = match self.previous().token_type {
                TokenType::EqualsEquals => Op::Equal,
                token_type => panic!(
                    "[line {}] Error: unexpected binary operator \'{:?}\'",
                    self.previous().line,
                    token_type
                ),
            };

            let right = self.comparison();

            equality = Ast::BinOp(Box::new(equality), operator, Box::new(right));
        }

        equality
    }

    fn access(&mut self) -> Ast {
        if DEBUG {
            println!("access(); {:?}", self.peek())
        };
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
                panic!(
                    "[line {}] Error: Expected at least one component when swizzling",
                    self.previous().line
                );
            }

            access = Ast::VecAccess(
                Box::new(access),
                Swizzle(
                    *accessors.get(0).unwrap(),
                    accessors.get(1).copied(),
                    accessors.get(2).copied(),
                    accessors.get(3).copied(),
                ),
            );
        }

        access
    }

    fn comparison(&mut self) -> Ast {
        if DEBUG {
            println!("comparison(); {:?}", self.peek())
        };
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

            term = Ast::BinOp(Box::new(term), op, Box::new(right));
        }

        term
    }

    fn term(&mut self) -> Ast {
        if DEBUG {
            println!("term(); {:?}", self.peek())
        };
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

            factor = Ast::BinOp(Box::new(factor), op, Box::new(right));
        }

        factor
    }

    fn factor(&mut self) -> Ast {
        if DEBUG {
            println!("factor(); {:?}", self.peek())
        };
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

            primary = Ast::BinOp(Box::new(primary), op, Box::new(right));
        }

        primary
    }

    fn primary(&mut self) -> Ast {
        if DEBUG {
            println!("primary(); {:?}", self.peek())
        };

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
            return Ast::Block(vec![Ast::Give(Box::new(expression))]);
        }

        if self.match_token(TokenType::Identifier) {
            return self.identity();
        }

        self.literal()
    }

    fn literal(&mut self) -> Ast {
        if DEBUG {
            println!("literal(); {:?}", self.peek())
        };
        // number literal
        if self.match_token(TokenType::Number) {
            return self.number();
        }

        self.vector()
    }

    fn vector(&mut self) -> Ast {
        if DEBUG {
            println!("vector(); {:?}", self.peek())
        };
        /*
            literal: [1; 2]
            semicolon: [1, 2, 3, 4]
        */

        self.consume(TokenType::LeftSquare, "Expected '[' in vector literal");

        let first = self.r#if();

        if self.match_token(TokenType::Semi) {
            self.consume(TokenType::Number, "Expected scalar literal as length in vector");

            let length = self.number();

            self.consume(
                TokenType::RightSquare,
                "vector literals must end with ']' and be up to 4 components",
            );

            return Ast::VecRepeated(Box::new(first), Box::new(length));
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

        Ast::VecLiteral(Box::new(first), Box::new(second), third, fourth)
    }

    fn number(&mut self) -> Ast {
        if DEBUG {
            println!("number(); {:?}", self.peek())
        };

        Ast::V(Val::Float(self.previous().lexeme.parse::<f32>().unwrap()))
    }

    fn identity(&mut self) -> Ast {
        if DEBUG {
            println!("identity(); {:?}", self.peek())
        };
        Ast::Ident(self.rodeo.get_or_intern(self.previous().lexeme))
    }

    fn call(&mut self) -> Ast {
        if DEBUG {
            println!("call(); {:?}", self.peek())
        };
        let token = self.previous();
        let name = &token.lexeme;

        // TODO: error handling
        let builtin = *self.builtins.get(name).expect(
            format!(
                "[line {}] Error: built-in function '{}' does not exist",
                token.line, name
            )
            .as_str(),
        );

        println!("BUILT-IN, GOT {:?}", builtin);

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

        let result = Ast::Call(builtin, arguments);

        println!("RESULT OF CALL: {:?}", result);
        println!("Current token: {:?}", self.peek());

        result
    }

    fn block(&mut self) -> Ast {
        if DEBUG {
            println!("block(); {:?}", self.peek())
        };
        let mut statements = Vec::new();

        while !self.match_token(TokenType::RightBracket) {
            if self.match_token(TokenType::Give) {
                statements.push(self.give());
                println!("Pushing give: {:?}", self.peek());
            } else {
                let statement = self.statement();

                println!("PUSHING STATEMENT TO BLOCK: {:?}", statement);
                statements.push(statement);
            }
        }

        let block = Ast::Block(statements);

        println!("FINISHED BLOCK: {:?}", block);

        block
    }

    fn give(&mut self) -> Ast {
        if DEBUG {
            println!("give(); {:?}", self.peek())
        };
        // wrap expression for give statement
        let expression = self.r#if();

        Ast::Give(Box::new(expression))
    }

    fn match_token(&mut self, token_type: TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            return true;
        }

        false
    }

    fn identifier(&mut self, name: String) -> Spur {
        if DEBUG {
            println!("identifier(); {:?}", self.peek())
        };
        println!("IDentifier: '{}'", name);
        self.rodeo.get_or_intern(name)
    }

    fn consume(&mut self, token_type: TokenType, error: &str) -> Token {
        if self.peek().token_type == token_type {
            return self.advance();
        }

        println!("End: {:?}; {:?}", self.peek(), self.lines);
        panic!("[line {}] Error: {}", self.peek().line, error);
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
