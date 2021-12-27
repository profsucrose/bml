use std::collections::HashMap;

use crate::{token::Token, token_type::TokenType};

pub struct Macro {
    keys: Vec<String>,
    template: Vec<Token>,
}

impl Macro {
    pub fn new(keys: Vec<String>, template: Vec<Token>) -> Macro {
        Macro { keys, template }
    }

    pub fn expand(&self, args: Vec<Vec<Token>>) -> Vec<Token> {
        // dot(a, b)
        let symbols = self
            .keys
            .clone()
            .into_iter()
            .zip(args.into_iter())
            .collect::<HashMap<_, _>>();

        self.template
            .clone()
            .into_iter()
            .flat_map(|t| match (t.token_type, symbols.get(&t.lexeme)) {
                (TokenType::Identifier, Some(expansion)) => expansion.clone(),
                _ => vec![t],
            })
            .collect::<Vec<Token>>()
    }
}
