use std::collections::HashMap;

use crate::{scanner::Scanner, token::Token, token_type::TokenType};

// macro dot(a, b) a.x * b.x + a.y * b.y
// (r, b) r.r * b.x + a.y * b.y
// dot([0, 1] + [1, 1], [2, 2]) a.r * b.x + a.y * b.y

pub struct Macro {
    keys: Vec<String>,
    template: Vec<(TokenType, String)>,
}

// TODO: add string interning
impl Macro {
    pub fn new(keys: Vec<String>, template: Vec<(TokenType, String)>) -> Macro {
        Macro { keys, template }
    }

    pub fn expand(&self, line: usize, args: Vec<Vec<Token>>) -> Vec<Token> {
        // dot(a, b)
        let symbols = self.keys.clone().into_iter().zip(args.into_iter()).collect::<HashMap<_, _>>();

        self.template.clone().into_iter().flat_map(
                |(t, s)| match (t, symbols.get(&s)) {
                    (TokenType::Identifier, Some(expansion)) => expansion.clone(),
                    _ => vec![Token::new(t, s, line)]
                }
            )
            .collect::<Vec<Token>>()
    }
}
