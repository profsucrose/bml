use bml::{r#macro::Macro, token::Token, token_type::TokenType};

pub fn main() {
    println!("Macro testing");

    // macro dot(a, b) a.x * b.x + a.y + b.y
    let dot = Macro::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            (TokenType::Identifier, "a".to_string()),
            (TokenType::Dot, ".".to_string()),
            (TokenType::X, "x".to_string()),
            (TokenType::Star, "*".to_string()),
            (TokenType::Identifier, "b".to_string()),
            (TokenType::Dot, ".".to_string()),
            (TokenType::X, "x".to_string()),
            (TokenType::Plus, "+".to_string()),
            (TokenType::Identifier, "a".to_string()),
            (TokenType::Dot, ".".to_string()),
            (TokenType::Y, "y".to_string()),
            (TokenType::Identifier, "b".to_string()),
            (TokenType::Dot, ".".to_string()),
            (TokenType::Y, "y".to_string()),
        ],
    );

    let vec = vec![
        Token::new(TokenType::LeftSquare, "[".to_string(), 1),
        Token::new(TokenType::Number, "1.0".to_string(), 1),
        Token::new(TokenType::Semi, ";".to_string(), 1),
        Token::new(TokenType::Number, "2".to_string(), 1),
        Token::new(TokenType::RightSquare, "]".to_string(), 1),
    ];

    let result = dot.expand(1, vec![vec.clone(), vec]);

    println!(
        "{:?}",
        result.into_iter().map(|t| t.lexeme).collect::<String>()
    );
}
