use std::collections::HashMap;

use crate::{scanner::Scanner, token::Token, token_type::TokenType};

// macro dot(a, b) a.x * b.x + a.y * b.y
// (r, b) r.r * b.x + a.y * b.y
// dot([0, 1] + [1, 1], [2, 2]) a.r * b.x + a.y * b.y

pub struct Macro {
    keys: Vec<String>,
    template: Vec<(TokenType, String)>
}

// TODO: add string interning
impl Macro {
    pub fn new(keys: Vec<String>, template: Vec<(TokenType, String)>) -> Macro {
        Macro { keys, template }
    }

    pub fn expand(&self, line: usize, args: Vec<Vec<Token>>) -> Vec<Token> {
        // dot(a, b)
        let mut symbols = HashMap::new();

        for i in 0..args.len() {
            symbols.insert(self.keys[i].to_owned(), args[i].to_owned());
        }

        let mut result = Vec::new();

        for token in self.template.iter() {
            if token.0 == TokenType::Identifier {
                if let Some(expansion) = symbols.get(&token.1) {
                    for t in expansion {
                        let t = Token::new(t.token_type, t.lexeme.clone(), t.line);
                        result.push(t.clone());
                    }

                    continue;
                }
            }

            result.push(Token::new(token.0, token.1.clone(), line));
        }

        result
    }
}